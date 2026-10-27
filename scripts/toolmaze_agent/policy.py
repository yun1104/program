"""API teacher and HF student backends. Heavy dependencies are imported lazily."""
from __future__ import annotations

import json

from .common import credential_secret
from .protocol import END_ACTION


class APIBackend:
    def __init__(self, config):
        self.config = config
        self.resolved_models = set()
        self.provider_usage = []
        self.protocol = config.get("protocol", "openai")
        if self.protocol == "anthropic":
            import httpx
            import uuid
            self._secret = credential_secret(config)
            self.client = httpx.Client(base_url=config["base_url"].rstrip("/") + "/",
                timeout=httpx.Timeout(config.get("timeout", 180), connect=20), trust_env=False,
                transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=0),
                headers={"Authorization": "Bearer " + self._secret, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json", "User-Agent": "ToolMazeTeacherCollection/1.0",
                         "x-opencode-session": config.get("session_id", "toolmaze-" + uuid.uuid4().hex)})
        elif self.protocol == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=credential_secret(config),
                                 base_url=config["base_url"], timeout=config.get("timeout", 120),
                                 max_retries=config.get("max_retries", 2))
        else:
            raise ValueError("API protocol must be openai or anthropic")

    def close(self):
        self.client.close()

    def initialize_episode(self, query, definitions):
        if not self.config.get("native_tool_calling"):
            return
        from .protocol import SYSTEM
        self.native_system = SYSTEM.split("At each <action>")[0] + (
            "\nUse the supplied native tool calls, one tool at a time. Wait for actual tool results. "
            "Never simulate tool outputs. When finished, give a concise grounded final answer. "
            "If a required operation cannot be completed, explicitly explain the limitation.")
        self.native_tools = []
        for definition in definitions:
            spec = definition.get("paradigms", {}).get("function_call", {}).get("spec", definition)
            self.native_tools.append({"name": spec["name"], "description": spec.get("description", ""),
                                      "input_schema": spec.get("parameters", {"type": "object", "properties": {}})})
        self.native_messages = [{"role": "user", "content": query}]
        self._pending_tool = None
        self._native_queue = []
        self._native_results = []

    def _next_native_action(self):
        from .protocol import encode_action
        if self._pending_tool is not None:
            raise RuntimeError("Previous native tool call has no actual result")
        call = self._native_queue.pop(0)
        self._pending_tool = call["id"]
        arguments = call["input"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return encode_action({"type": "tool_call", "tool_name": call["name"], "arguments": arguments})

    def receive_tool_result(self, name, result):
        if not self.config.get("native_tool_calling"):
            return
        if self._pending_tool is None:
            raise RuntimeError("Native tool result has no matching tool call")
        self._native_results.append({"type": "tool_result",
            "tool_use_id": self._pending_tool, "content": json.dumps(result, ensure_ascii=False),
            "is_error": isinstance(result, dict) and "error" in result})
        self._pending_tool = None
        if not self._native_queue:
            self.native_messages.append({"role": "user", "content": self._native_results})
            self._native_results = []

    def receive_protocol_error(self, message):
        if self.config.get("native_tool_calling") and self._pending_tool is not None:
            self.receive_tool_result("protocol", {"error": message})

    def _anthropic_generate(self, request):
        import httpx
        import time
        system = "\n\n".join(message["content"] for message in request if message["role"] == "system")
        payload = {"model": self.config["model"], "max_tokens": self.config.get("max_tokens", 1024),
                   "temperature": self.config.get("temperature", .8), "top_p": 1,
                   "stop_sequences": ["</action>"],
                   "thinking": {"type": "disabled"}, "stream": False,
                   "messages": [message for message in request if message["role"] != "system"]}
        if system:
            payload["system"] = system
        native = self.config.get("native_tool_calling", False) and hasattr(self, "native_messages")
        if native:
            payload.update(system=self.native_system, messages=self.native_messages,
                           tools=self.native_tools, tool_choice={"type": "auto", "disable_parallel_tool_use": True})
            payload.pop("stop_sequences", None)
        maximum = self.config.get("max_retries", 2)
        for attempt in range(maximum + 1):
            try:
                response = self.client.post("v1/messages", json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == maximum:
                    raise
                time.sleep(min(2 ** (attempt + 1), 10))
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < maximum:
                try:
                    delay = float(response.headers.get("retry-after", 2 ** (attempt + 1)))
                except ValueError:
                    delay = 2 ** (attempt + 1)
                time.sleep(max(0, min(delay, 30)))
                continue
            if response.status_code != 200:
                message = response.text.replace(self._secret, "<REDACTED>")[:800]
                raise RuntimeError(f"Provider HTTP {response.status_code}: {message}")
            try:
                data = response.json()
            except ValueError as error:
                raise RuntimeError("Provider returned a non-JSON response with HTTP 200") from error
            self.resolved_models.add(data.get("model", self.config["model"]))
            blocks = data.get("content", [])
            if any(block.get("type") not in ({"text", "tool_use"} if native else {"text"}) for block in blocks):
                raise RuntimeError("Unexpected non-text block with thinking disabled and text-action protocol")
            if native:
                from .protocol import encode_action
                self.native_messages.append({"role": "assistant", "content": blocks})
                calls = [block for block in blocks if block["type"] == "tool_use"]
                if calls:
                    # Some compatible gateways ignore disable_parallel_tool_use. Execute every
                    # returned call once, then send one complete batch of actual tool results.
                    self._native_queue = list(calls)
                    self._native_results = []
                    text = self._next_native_action()
                else:
                    content = "".join(block.get("text", "") for block in blocks)
                    text = encode_action({"type": "final_answer", "content": content}) if content else ""
            else:
                text = "".join(block.get("text", "") for block in blocks)
            if not text:
                raise RuntimeError("Teacher returned no text content")
            native_usage = data.get("usage", {})
            self.provider_usage.append(native_usage)
            return {"text": text, "usage": {
                "input_tokens": sum(native_usage.get(key, 0) or 0 for key in
                    ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
                "output_tokens": native_usage.get("output_tokens", 0) or 0},
                "finish_reason": data.get("stop_reason"), "provider_request_id": data.get("id"),
                "provider_stop_sequence": data.get("stop_sequence"),
                "provider_content_blocks": blocks,
                "provider_usage": native_usage}
        raise RuntimeError("Provider retry loop exhausted")

    def generate(self, messages, segments, **kwargs):
        if self.config.get("native_tool_calling") and getattr(self, "_native_queue", []):
            return {"text": self._next_native_action(), "usage": {"input_tokens": 0, "output_tokens": 0},
                    "finish_reason": "queued_native_tool_use"}
        transcript = "".join(segment["text"] for segment in segments)
        request = [dict(message) for message in messages]
        if transcript:
            request[-1]["content"] += transcript
        if self.protocol == "anthropic":
            return self._anthropic_generate(request)
        response = self.client.chat.completions.create(
            model=self.config["model"], messages=request,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 512),
            stop=["</action>"],
            extra_body=self.config.get("extra_body", {}))
        self.resolved_models.add(response.model)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Teacher returned no content; check thinking mode and output token budget")
        return {"text": content, "usage": {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0},
            "finish_reason": response.choices[0].finish_reason}


def load_student(model_path, dtype="bfloat16", freeze_vision=True):
    from pathlib import Path
    from .common import project_path
    if Path(model_path).is_absolute() or str(model_path).startswith(("models/", "runs/", "./", "../")):
        checkpoint = project_path(model_path)
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing local model checkpoint: {checkpoint}. Download weights with tools/bootstrap.py --model or set model to an existing checkpoint.")
        model_path = str(checkpoint)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path)
    cls = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
    model = cls.from_pretrained(model_path, dtype=getattr(torch, dtype), attn_implementation="sdpa")
    # Freeze visual modules only. The language backbone, embeddings and LM head remain trainable.
    frozen = []
    if freeze_vision:
        for name, parameter in model.named_parameters():
            if any(part in {"visual", "vision_tower", "vision_model", "vision_encoder"} for part in name.split(".")):
                parameter.requires_grad_(False)
                frozen.append(name)
    model.config.use_cache = False
    return model, tokenizer, frozen


class HFBackend:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config

    def generate(self, messages, segments, input_ids=None, **kwargs):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from .protocol import token_segments
        tokenizer = self.tokenizer

        class EndAction(StoppingCriteria):
            def __call__(self, ids, scores, **unused):
                tail = tokenizer.decode(ids[0, start:].tolist(), skip_special_tokens=False)
                return tail.rstrip().endswith("</action>")

        if input_ids is None:
            input_ids, _ = token_segments(tokenizer, messages, segments)
        start = len(input_ids)
        context = self.config.get("max_context_tokens", 8192)
        # Keep one slot for the runtime terminal marker used by GRPO.
        max_new = min(self.config.get("max_tokens", 512), context - start - 1)
        if max_new <= 0:
            raise ValueError("Context budget exhausted; history is never silently truncated")
        device = next(self.model.parameters()).device
        tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        temperature = self.config.get("temperature", 0.8)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    max_new_tokens=max_new, do_sample=temperature > 0,
                    **({"temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature > 0 else {}),
                    stopping_criteria=StoppingCriteriaList([EndAction()]),
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    use_cache=True)
        finally:
            self.model.train(was_training)
        generated = output[0, start:].tolist()
        return {"text": tokenizer.decode(generated, skip_special_tokens=True),
                "token_ids": generated, "usage": {"input_tokens": start, "output_tokens": len(generated)}}
