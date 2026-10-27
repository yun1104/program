"""Run the same environment/policy loop for collection, evaluation and GRPO."""
from __future__ import annotations

from .environment import ToolMazeEnvironment
from .protocol import encode_action, initial_messages, observation_text, parse_action, token_segments


def run_episode(task, backend, config):
    env = ToolMazeEnvironment(task, config)
    if hasattr(backend, "initialize_episode"):
        backend.initialize_episode(env.query, env.definitions)
    messages = initial_messages(env.query, env.definitions)
    segments = []
    decisions = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    invalid = 0
    errors = []
    reason = "max_rounds"

    def append_observation(name, value):
        segment = {"text": observation_text(name, value), "model": False}
        if hasattr(backend, "tokenizer"):
            ids, _ = token_segments(backend.tokenizer, messages, segments + [segment])
            if len(ids) + 1 > backend.config.get("max_context_tokens", 8192):
                errors.append("Observation exceeds remaining context budget; episode terminated without truncation")
                return False
        segments.append(segment)
        return True

    for _ in range(config.get("max_rounds", 16)):
        try:
            response = backend.generate(messages, segments)
        except ValueError as error:
            errors.append(str(error))
            reason = "generation_error"
            break
        for key in usage:
            usage[key] += response.get("usage", {}).get(key, 0)
        text = response["text"]
        provider = {key: response[key] for key in ("finish_reason", "provider_request_id", "provider_usage", "provider_stop_sequence", "provider_content_blocks")
                    if key in response}
        try:
            action = parse_action(text, env.allowed)
        except (ValueError, TypeError) as error:
            invalid += 1
            # Raw malformed actions remain model tokens; feedback remains external tokens.
            segments.append({"text": text, "model": True, **({"provider": provider} if provider else {}), **({"token_ids": response["token_ids"]}
                            if "token_ids" in response else {})})
            if "sampling_logprobs" in response:
                segments[-1]["sampling_logprobs"] = response["sampling_logprobs"]
            decisions.append({"valid": False})
            if hasattr(backend, "receive_protocol_error"):
                backend.receive_protocol_error(str(error))
            if not append_observation("protocol", {"error": str(error)}):
                reason = "context_budget"
                break
            continue
        segment = {"text": text if "token_ids" in response else encode_action(action), "model": True}
        if provider:
            segment["provider"] = provider
        if "token_ids" not in response:
            segment["raw_response_text"] = text
        if "token_ids" in response:
            segment["token_ids"] = response["token_ids"]
        if "sampling_logprobs" in response:
            segment["sampling_logprobs"] = response["sampling_logprobs"]
        segments.append(segment)
        decisions.append({"valid": True, "type": action["type"]})
        result = env.step(action)
        if action["type"] == "final_answer":
            reason = "final_answer"
            break
        if hasattr(backend, "receive_tool_result"):
            backend.receive_tool_result(action["tool_name"], result)
        if not append_observation(action["tool_name"], result):
            reason = "context_budget"
            break
    usage["total_tokens"] = sum(usage.values())
    trace, judgement = env.score(usage)
    strict_pass = bool(judgement["pass"] and env.finished and invalid == 0 and env.validation_errors == 0)
    return {"initial_messages": messages, "segments": segments, "trace": trace,
            "judgement": judgement, "official_pass": bool(judgement["pass"]), "strict_pass": strict_pass,
            "termination": reason, "invalid_actions": invalid, "validation_errors": env.validation_errors,
            "tool_calls": env.calls, "tokens": usage, "errors": errors}


def flatten_episode(episode, tokenizer):
    prompt, _ = token_segments(tokenizer, episode["initial_messages"], [])
    completion, mask = [], []
    for segment in episode["segments"]:
        ids = segment.get("token_ids")
        if ids is None:
            ids = tokenizer.encode(segment["text"], add_special_tokens=False)
        completion.extend(ids)
        mask.extend([int(segment["model"])] * len(ids))
    if not completion or not any(mask):
        raise ValueError("Rollout has no model tokens")
    # The terminal marker is inserted by the runtime and carries no policy gradient.
    if completion[-1] != tokenizer.eos_token_id:
        completion.append(tokenizer.eos_token_id)
        mask.append(0)
    return prompt, completion, mask


def episode_reward(episode, call_penalty=0.002):
    # Cost only discriminates successful trajectories. Failing early cannot beat succeeding.
    if not episode["strict_pass"]:
        return -0.1 * min(episode["invalid_actions"] + episode["validation_errors"], 5)
    return 1.0 - min(call_penalty * episode["tool_calls"], 0.2)
