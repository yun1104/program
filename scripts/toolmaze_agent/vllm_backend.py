"""Batched vLLM generation using the same exact token protocol as HF training."""
from __future__ import annotations

from concurrent.futures import Future
from queue import Empty, Queue
import time

from .common import fingerprint
from .protocol import token_segments


class VLLMBackend:
    def __init__(self, model_path, generation, engine, seed):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = generation
        self.engine = engine
        self.seed = seed
        self.requests = Queue()
        self.fatal_error = None
        self.batch_counter = 0
        extra = {}
        if engine.get("compilation_config") is not None:
            extra["compilation_config"] = engine["compilation_config"]
        self.llm = LLM(model=model_path, dtype="bfloat16", seed=seed,
                       generation_config="vllm", max_model_len=generation["max_context_tokens"],
                       tensor_parallel_size=engine.get("tensor_parallel_size", 1),
                       gpu_memory_utilization=engine.get("gpu_memory_utilization", 0.65),
                       enforce_eager=engine.get("enforce_eager", True),
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0},
                       max_num_seqs=engine.get("max_num_seqs", 16),
                       max_num_batched_tokens=engine.get("max_num_batched_tokens", 8192),
                       disable_log_stats=True, **extra)

    def session(self, identity):
        return VLLMSession(self, identity)

    def generate_batch(self, requests):
        from vllm import SamplingParams
        prompts, params = [], []
        for request in requests:
            ids, _ = token_segments(self.tokenizer, request["messages"], request["segments"])
            max_new = min(self.config["max_tokens"], self.config["max_context_tokens"] - len(ids) - 1)
            if max_new <= 0:
                raise ValueError("Context budget exhausted; history is never silently truncated")
            prompts.append({"prompt_token_ids": ids})
            params.append(SamplingParams(temperature=self.config["temperature"], top_p=1.0, top_k=-1,
                         max_tokens=max_new, stop=["</action>"], include_stop_str_in_output=True,
                         seed=request["seed"]))
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [{"text": output.outputs[0].text, "token_ids": list(output.outputs[0].token_ids),
                 "finish_reason": output.outputs[0].finish_reason,
                 "usage": {"input_tokens": len(prompt["prompt_token_ids"]),
                           "output_tokens": len(output.outputs[0].token_ids)}}
                for output, prompt in zip(outputs, prompts)]

    def dispatch_until_done(self, futures):
        """Only this coordinator thread calls LLM.generate; workers only run environments."""
        if self.engine.get("continuous_batching"):
            return self.dispatch_continuous(futures)
        while not all(future.done() for future in futures):
            try:
                first = self.requests.get(timeout=0.05)
            except Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + 0.02
            while len(batch) < self.engine.get("max_num_seqs", 16):
                try:
                    batch.append(self.requests.get(timeout=max(0, deadline - time.monotonic())))
                except Empty:
                    break
            if self.fatal_error:
                for request in batch:
                    request["future"].set_exception(RuntimeError(self.fatal_error))
                continue
            try:
                started = time.monotonic()
                outputs = self.generate_batch(batch)
                if len(outputs) != len(batch):
                    raise RuntimeError("vLLM returned a different number of outputs than requested")
                for request, output in zip(batch, outputs):
                    request["future"].set_result(output)
                self.batch_counter += 1
                print(f"vLLM batch {self.batch_counter}: requests={len(batch)}, "
                      f"generated_tokens={sum(item['usage']['output_tokens'] for item in outputs)}, "
                      f"seconds={time.monotonic() - started:.2f}", flush=True)
            except Exception as error:
                self.fatal_error = f"vLLM generation failed: {type(error).__name__}: {error}"
                for request in batch:
                    request["future"].set_exception(RuntimeError(self.fatal_error))

    def dispatch_continuous(self, futures):
        """Return each finished action immediately; never wait for the slowest action in a batch."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        core = self.llm.llm_engine
        active = {}
        serial = 0
        last_log = time.monotonic()
        finished_count, generated_tokens = 0, 0
        while not all(future.done() for future in futures) or active:
            while len(active) < self.engine["max_num_seqs"]:
                try:
                    request = self.requests.get_nowait()
                except Empty:
                    break
                if self.fatal_error:
                    request["future"].set_exception(RuntimeError(self.fatal_error))
                    continue
                try:
                    ids = request["prompt_ids"]
                    max_new = min(self.config["max_tokens"], self.config["max_context_tokens"] - len(ids) - 1)
                    params = SamplingParams(temperature=self.config["temperature"], top_p=1.0, top_k=-1,
                        max_tokens=max_new, stop=["</action>"], include_stop_str_in_output=True,
                        seed=request["seed"], output_kind=RequestOutputKind.FINAL_ONLY)
                    # add_request returns an internal randomized ID, but RequestOutput uses the external ID.
                    key = f"toolmaze-{serial}"
                    core.add_request(key, {"prompt_token_ids": ids}, params)
                    serial += 1
                    active[key] = request
                except Exception as error:
                    request["future"].set_exception(error)
            if not active:
                time.sleep(0.002)
                continue
            try:
                for output in core.step():
                    if not output.finished:
                        continue
                    request = active.pop(output.request_id)
                    completion = output.outputs[0]
                    request["future"].set_result({"text": completion.text,
                        "token_ids": list(completion.token_ids), "finish_reason": completion.finish_reason,
                        "usage": {"input_tokens": len(request["prompt_ids"]),
                                  "output_tokens": len(completion.token_ids)}})
                    finished_count += 1
                    generated_tokens += len(completion.token_ids)
            except Exception as error:
                self.fatal_error = f"vLLM continuous engine failed: {type(error).__name__}: {error}"
                for request in active.values():
                    request["future"].set_exception(RuntimeError(self.fatal_error))
                active.clear()
            now = time.monotonic()
            if now - last_log >= 10:
                print(f"vLLM continuous: active={len(active)}, completed_actions={finished_count}, "
                      f"completed_output_tokens={generated_tokens}", flush=True)
                last_log = now


class VLLMSession:
    def __init__(self, backend, identity):
        self.backend = backend
        self.config, self.tokenizer = backend.config, backend.tokenizer
        self.identity = identity
        self.counter = 0

    def generate(self, messages, segments):
        if self.backend.fatal_error:
            raise RuntimeError(self.backend.fatal_error)
        # Check locally so one context-exhausted episode cannot fail unrelated batch items.
        ids, _ = token_segments(self.tokenizer, messages, segments)
        if len(ids) + 1 >= self.config["max_context_tokens"]:
            raise ValueError("Context budget exhausted; history is never silently truncated")
        seed = int(fingerprint([self.backend.seed, self.identity, self.counter])[:8], 16)
        self.counter += 1
        future = Future()
        self.backend.requests.put({"messages": messages, "segments": list(segments), "seed": seed,
                                   "prompt_ids": ids, "future": future})
        return future.result()
