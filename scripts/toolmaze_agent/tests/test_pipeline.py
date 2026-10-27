import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.toolmaze_agent.common import ROOT, digest, load_config, read_json, read_rows, write_json, write_rows
from scripts.toolmaze_agent.environment import ToolMazeEnvironment
from scripts.toolmaze_agent.prepare import grouped_rows
from scripts.toolmaze_agent.protocol import encode_action, initial_messages, parse_action, token_segments
from scripts.toolmaze_agent.rollout import episode_reward, flatten_episode, run_episode


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [byte + 1 for byte in text.encode()]

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False) + "\nassistant:\n"


class ScriptedPolicy:
    """Test oracle only; never used by collection or training commands."""
    def __init__(self, actions):
        self.actions = iter(actions)

    def generate(self, messages, segments):
        return {"text": encode_action(next(self.actions)), "usage": {"input_tokens": 1, "output_tokens": 1}}


def action(step):
    return {"type": "tool_call", "tool_name": step["tool_name"], "arguments": copy.deepcopy(step["arguments"])}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = read_json(ROOT / "data/toolmaze/perturbed_tasks/c1/C1_task_001_P0.json")

    def test_split_isolation_and_determinism(self):
        rows = grouped_rows(ROOT / "data/toolmaze/perturbed_tasks", 123, [.8, .1, .1])
        self.assertEqual(len(rows), 2000)
        for field in ("group", "template_id", "task_id"):
            assignment = {}
            for row in rows:
                self.assertEqual(assignment.setdefault(row[field], row["split"]), row["split"])
        self.assertEqual(rows, grouped_rows(ROOT / "data/toolmaze/perturbed_tasks", 123, [.8, .1, .1]))

    def test_schema_public_view_and_hidden_metadata(self):
        env = ToolMazeEnvironment(self.task, {})
        prompt = json.dumps(initial_messages(env.query, env.definitions))
        for private in ("perturbation_mode", "perturbation_status", "expected_result", "valid_paths", "execution_trace"):
            self.assertNotIn(private, prompt)
        self.assertIn('parameters', prompt)
        self.assertIn('flight_number', prompt)

    def test_schema_error_does_not_trigger_fault(self):
        task = read_json(ROOT / "data/toolmaze/perturbed_tasks/c1/C1_task_001_P1.json")
        env = ToolMazeEnvironment(task, {})
        result = env.step({"type": "tool_call", "tool_name": "get_flight_info", "arguments": {}})
        self.assertIn("error", result)
        self.assertEqual(env.validation_errors, 1)
        self.assertEqual(len(env.engine.context.history), 0)

    def test_p0_end_to_end_actual_plugins_and_judge(self):
        actions = [action(step) for step in self.task["execution_trace"]]
        actions.append({"type": "final_answer", "content": "The simulated flight and hotel bookings were completed."})
        result = run_episode(self.task, ScriptedPolicy(actions), {})
        self.assertTrue(result["official_pass"], result["judgement"])
        self.assertTrue(result["strict_pass"])
        self.assertGreater(episode_reward(result), .8)
        tokens, labels = token_segments(ByteTokenizer(), result["initial_messages"], result["segments"])
        self.assertEqual(len(tokens), len(labels))
        self.assertGreater(sum(label == -100 for label in labels), 0)
        prompt, completion, mask = flatten_episode(result, ByteTokenizer())
        self.assertEqual(len(completion), len(mask))
        self.assertEqual(mask[-1], 0)  # runtime terminal marker
        self.assertEqual(sum(mask), sum(label != -100 for label in labels))

    def test_missing_final_is_not_strict_success(self):
        actions = [action(step) for step in self.task["execution_trace"]]
        result = run_episode(self.task, ScriptedPolicy(actions), {"max_rounds": len(actions)})
        self.assertTrue(result["official_pass"])
        self.assertFalse(result["strict_pass"])

    def test_transient_fault_resets_per_episode(self):
        task = read_json(ROOT / "data/toolmaze/perturbed_tasks/c1/C1_task_001_P1.json")
        for _ in range(2):
            env = ToolMazeEnvironment(task, {})
            victim = next(iter(env.engine.perturbation_map))
            step = next(step for step in task["execution_trace"] if step["tool_name"] == victim)
            call = action(step)
            env.step(call)
            env.step(call)
            messages = [m for m in env.engine.logger.messages if m["role"] == "tool"]
            self.assertEqual([m["metadata"]["perturbation_status"] for m in messages], ["perturbed", "clean"])

    def test_sampled_ids_are_preserved(self):
        segments = [{"text": "different decoded text", "model": True, "token_ids": [11, 12]},
                    {"text": "observation", "model": False}]
        ids, labels = token_segments(ByteTokenizer(), [], segments)
        prefix = len(ByteTokenizer().encode(ByteTokenizer().apply_chat_template([])))
        self.assertEqual(ids[prefix:prefix + 2], [11, 12])
        self.assertEqual(labels[prefix:prefix + 2], [11, 12])
        self.assertTrue(all(x == -100 for x in labels[prefix + 2:]))

    def test_permanent_fault_and_alternative(self):
        task = read_json(ROOT / "data/toolmaze/perturbed_tasks/c2/C2_task_001_P2.json")
        env = ToolMazeEnvironment(task, {})
        alternatives = env.engine.alternative_tools
        calls = {}
        for path in task["valid_paths"]:
            for step in path["execution_trace"]:
                if step["tool_name"] in alternatives:
                    calls[step["tool_name"]] = action(step)
        first, second = list(calls)[:2]
        env.step(calls[first])
        env.step(calls[first])
        env.step(calls[second])
        observations = [m for m in env.engine.logger.messages if m["role"] == "tool"]
        self.assertEqual([m["metadata"]["perturbation_status"] for m in observations],
                         ["perturbed", "perturbed", "clean"])

    def test_infrastructure_errors_remain_in_denominator(self):
        from scripts.toolmaze_agent.eval import summarize
        path = ROOT / "data/toolmaze/perturbed_tasks/c1/C1_task_001_P0.json"
        row = {"id": "C1_task_001:P0", "mode": "P0", "complexity": "C1", "domains": ["Travel"],
               "path": str(path), "sha256": digest(path)}
        result = run_episode(self.task, ScriptedPolicy([action(s) for s in self.task["execution_trace"]] + [
            {"type": "final_answer", "content": "Done."}]), {})
        report = summarize([{"row": row, **result}, {"row": row, "error": "offline"}],
                           {"upstream": "tools/ToolMaze"})
        self.assertEqual(report["breakdown"]["all"]["n"], 2)
        self.assertEqual(report["breakdown"]["all"]["official_pass_rate"], .5)
        self.assertEqual(report["breakdown"]["all"]["infrastructure_errors"], 1)

    def test_unknown_tool_and_non_json_rejected(self):
        for text in ('{"type":"tool_call","tool_name":"other","arguments":{}}', 'not JSON'):
            with self.assertRaises(ValueError):
                parse_action(text, {"get_flight_info"})

    def test_context_limit_stops_before_oversized_observation(self):
        tokenizer = ByteTokenizer()
        env = ToolMazeEnvironment(self.task, {})
        messages = initial_messages(env.query, env.definitions)
        text = encode_action(action(self.task["execution_trace"][0]))
        prefix, _ = token_segments(tokenizer, messages, [])
        maximum = len(prefix) + len(tokenizer.encode(text)) + 1

        class LimitedPolicy:
            config = {"max_context_tokens": maximum}

            def generate(self, messages, segments):
                return {"text": text, "token_ids": tokenizer.encode(text)}

        backend = LimitedPolicy()
        backend.tokenizer = tokenizer
        episode = run_episode(self.task, backend, {})
        self.assertEqual(episode["termination"], "context_budget")
        self.assertFalse(episode["strict_pass"])
        prompt, completion, mask = flatten_episode(episode, tokenizer)
        self.assertLessEqual(len(prompt) + len(completion), maximum)

    def test_collection_resume_and_sft_builder_without_api(self):
        from scripts.toolmaze_agent import collect, build_sft
        import yaml
        frozen = read_rows(ROOT / "data/splits/toolmaze_v1/train.jsonl")
        row = next(r for r in frozen if r["id"] == "C1_task_001:P0")
        actions = [action(step) for step in self.task["execution_trace"]]
        actions.append({"type": "final_answer", "content": "Simulated bookings completed."})

        class OfflineTeacher(ScriptedPolicy):
            calls = 0

            def __init__(self, config):
                super().__init__(actions)
                self.resolved_models = {"test-oracle-not-an-api-model"}

            def generate(self, *args, **kwargs):
                OfflineTeacher.calls += 1
                return super().generate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            splits = directory / "splits"
            write_rows(splits / "train.jsonl", [row])
            config = load_config("scripts/toolmaze_agent/configs/project.yaml")
            config["splits"] = str(splits)
            config_path = directory / "project.yaml"
            config_path.write_text(yaml.safe_dump(config))
            args = ["collect", "--config", str(config_path), "--output", str(directory / "collection"),
                    "--attempts", "1", "--workers", "1"]
            with patch("scripts.toolmaze_agent.collect.APIBackend", OfflineTeacher), patch("sys.argv", args):
                collect.main()
                calls = OfflineTeacher.calls
                collect.main()
                self.assertEqual(OfflineTeacher.calls, calls)  # completed episodes are not resampled
            with patch("sys.argv", ["build_sft", "--collection", str(directory / "collection"),
                                    "--output", str(directory / "sft")]):
                build_sft.main()
            rows = read_rows(directory / "sft/train.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], row["id"])


if __name__ == "__main__":
    unittest.main()
