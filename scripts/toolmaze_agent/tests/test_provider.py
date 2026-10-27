import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from scripts.toolmaze_agent.common import credential_secret
from scripts.toolmaze_agent.policy import APIBackend


class ProviderTests(unittest.TestCase):
    def config(self):
        return {"protocol": "anthropic", "model": "deepseek-v4.1-flash",
                "base_url": "https://opencode.ai/inference/go/anthropic", "api_key_env": "TEST_OPENCODE_KEY",
                "max_retries": 0, "temperature": .8, "max_tokens": 1024}

    def test_messages_mapping_stop_and_usage(self):
        observed = []

        def handle(request):
            observed.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://opencode.ai/inference/go/anthropic/v1/messages")
            self.assertEqual(request.headers["Authorization"], "Bearer oc-unit-test-only")
            return httpx.Response(200, json={"id": "unit-response", "model": "deepseek-v4.1-flash",
                "content": [{"type": "text", "text": '{"type":"final_answer","content":"done"}'}],
                "stop_reason": "stop_sequence", "stop_sequence": "</action>",
                "usage": {"input_tokens": 30, "cache_read_input_tokens": 70, "output_tokens": 10}})

        with patch.dict(os.environ, {"TEST_OPENCODE_KEY": "oc-unit-test-only"}):
            backend = APIBackend(self.config())
        backend.client.close()
        backend.client = httpx.Client(base_url=self.config()["base_url"] + "/",
                                     headers={"Authorization": "Bearer oc-unit-test-only"},
                                     transport=httpx.MockTransport(handle))
        try:
            result = backend.generate([{"role": "system", "content": "system rules"},
                                       {"role": "user", "content": "task"}], [])
        finally:
            backend.close()
        self.assertEqual(observed[0]["model"], "deepseek-v4.1-flash")
        self.assertEqual(observed[0]["system"], "system rules")
        self.assertEqual(observed[0]["stop_sequences"], ["</action>"])
        self.assertEqual(observed[0]["thinking"], {"type": "disabled"})
        self.assertEqual([m["role"] for m in observed[0]["messages"]], ["user"])
        self.assertEqual(result["usage"], {"input_tokens": 100, "output_tokens": 10})
        self.assertEqual(result["provider_stop_sequence"], "</action>")

    def test_credential_file_is_data_not_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "credentials"
            path.write_text("API_KEY='oc-unit-$(not-a-command)'\n")
            self.assertEqual(credential_secret({"api_key_env": "UNUSED_TEST_KEY", "credentials_file": str(path)}),
                             "oc-unit-$(not-a-command)")

    def test_native_parallel_calls_wait_for_actual_results(self):
        requests = []

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                blocks = [{"type": "tool_use", "id": "a", "name": "first", "input": {"x": 1}},
                          {"type": "tool_use", "id": "b", "name": "second", "input": {"x": 2}}]
                reason = "tool_use"
            else:
                results = body["messages"][-1]["content"]
                self.assertEqual([r["tool_use_id"] for r in results], ["a", "b"])
                self.assertEqual([json.loads(r["content"]) for r in results], [{"actual": 10}, {"actual": 20}])
                blocks, reason = [{"type": "text", "text": "Done."}], "end_turn"
            return httpx.Response(200, json={"id": "native-response", "model": "deepseek-v4.1-flash",
                "content": blocks, "stop_reason": reason, "usage": {"input_tokens": 20, "output_tokens": 10}})

        with patch.dict(os.environ, {"TEST_OPENCODE_KEY": "oc-unit-test-only"}):
            backend = APIBackend({**self.config(), "native_tool_calling": True})
        backend.client.close()
        backend.client = httpx.Client(base_url=self.config()["base_url"] + "/", transport=httpx.MockTransport(handle))
        definitions = [{"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}
                       for name in ("first", "second")]
        backend.initialize_episode("task", definitions)
        try:
            first = backend.generate([], [])
            self.assertIn('"tool_name":"first"', first["text"])
            backend.receive_tool_result("first", {"actual": 10})
            second = backend.generate([], [])
            self.assertEqual(len(requests), 1)
            self.assertEqual(second["usage"]["output_tokens"], 0)
            self.assertIn('"tool_name":"second"', second["text"])
            backend.receive_tool_result("second", {"actual": 20})
            final = backend.generate([], [])
            self.assertIn('"type":"final_answer"', final["text"])
            self.assertEqual(len(requests), 2)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
