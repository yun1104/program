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
        self.llm = LLM(model=model_path, dtype="bfloat16", seed=seed,
                       generation_config="vllm", max_model_len=generation["max_context_tokens"],
                       tensor_parallel_size=engine.get("tensor_parallel_size", 1),
                       gpu_memory_utilization=engine.get("gpu_memory_utilization", 0.65),
                       enforce_eager=engine.get("enforce_eager", True),
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0},
                       max_num_seqs=engine.get("max_num_seqs", 16),
                       disable_log_stats=True)

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
                                   "future": future})
        return future.result()
