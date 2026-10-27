"""Check external request routing and immediate continuation without a GPU."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
import sys
import time
import types
import unittest
from unittest.mock import patch

from scripts.toolmaze_agent.vllm_backend import VLLMBackend


class SchedulerTests(unittest.TestCase):
    def test_external_ids_and_no_batch_barrier(self):
        class Params:
            def __init__(self, **kwargs):
                self.values = kwargs

        class Core:
            def __init__(self):
                self.pending = {}
                self.tick = 0

            def add_request(self, identity, prompt, params):
                self.pending[identity] = self.tick + (50 if identity == "toolmaze-0" else 1)
                return identity + "-internal-random-suffix"

            def step(self):
                self.tick += 1
                time.sleep(.001)
                ready = [key for key, tick in self.pending.items() if tick <= self.tick]
                result = []
                for key in ready:
                    del self.pending[key]
                    result.append(types.SimpleNamespace(request_id=key, finished=True, outputs=[
                        types.SimpleNamespace(text=key, token_ids=[7], finish_reason="stop")]))
                return result

        backend = VLLMBackend.__new__(VLLMBackend)
        backend.config = {"temperature": .8, "max_tokens": 512, "max_context_tokens": 8192}
        backend.engine = {"max_num_seqs": 64}
        backend.requests = Queue()
        backend.fatal_error = None
        backend.llm = types.SimpleNamespace(llm_engine=Core())
        completion_order = []

        def worker(name, turns):
            for turn in range(turns):
                future = Future()
                backend.requests.put({"prompt_ids": [1, 2], "seed": 123, "future": future})
                output = future.result(timeout=5)
                completion_order.append((name, turn, output["text"]))

        fake = types.ModuleType("vllm")
        fake.SamplingParams = Params
        sampling = types.ModuleType("vllm.sampling_params")
        sampling.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="final")
        with patch.dict(sys.modules, {"vllm": fake, "vllm.sampling_params": sampling}):
            with ThreadPoolExecutor(max_workers=2) as executor:
                long = executor.submit(worker, "long", 1)
                while backend.requests.empty():
                    time.sleep(.001)
                short = executor.submit(worker, "short", 2)
                backend.dispatch_continuous([long, short])
                long.result()
                short.result()
        self.assertIsNone(backend.fatal_error)
        self.assertEqual([name for name, _, _ in completion_order], ["short", "short", "long"])
        self.assertEqual({text for _, _, text in completion_order}, {"toolmaze-0", "toolmaze-1", "toolmaze-2"})


if __name__ == "__main__":
    unittest.main()
