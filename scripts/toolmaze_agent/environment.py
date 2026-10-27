"""Adapter to the pinned ToolMaze runtime; private annotations stay here."""
from __future__ import annotations

import sys
from pathlib import Path

from .common import project_path


def upstream_modules(root):
    root = project_path(root).resolve()
    if not (root / "evaluation/core/sandbox.py").is_file():
        raise FileNotFoundError(f"Missing ToolMaze checkout at {root}; run tools/bootstrap.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evaluation.core.sandbox import ExecutionEngine
    from evaluation.core.judge import JudgeSystem
    from evaluation.core.metrics import MetricsCalculator
    return ExecutionEngine, JudgeSystem, MetricsCalculator


class ToolMazeEnvironment:
    def __init__(self, task, config, judge_client=None):
        Engine, Judge, _ = upstream_modules(config.get("upstream", "tools/ToolMaze"))
        root = project_path(config.get("upstream", "tools/ToolMaze"))
        # We deliberately do not use Engine.run(), which may set the agent's perturbation_mode.
        self.engine = Engine(task, agent=None, tools_dir=str(root / "tools"))
        self.task = task
        self.definitions = sorted(self.engine.tool_definitions, key=lambda tool: tool["name"])
        self.query = self.engine.task_description
        self.allowed = {tool["name"] for tool in self.definitions}
        required = {step["tool_name"] for step in self.engine.execution_trace if step.get("tool_name")}
        if required - self.allowed:
            raise ValueError(f"Missing public tool definitions: {sorted(required - self.allowed)}")
        template_dir = project_path(config.get("templates", "data/toolmaze/templates"))
        self.judge = Judge(llm_client=judge_client,
                           model=config.get("judge", {}).get("model", "deepseek-flash"),
                           temperature=0, template_dir=str(template_dir))
        self.engine.logger.log_user_message(self.query)
        self.calls = 0
        self.validation_errors = 0
        self.finished = False

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode already finished")
        if action["type"] == "final_answer":
            self.finished = True
            self.engine.logger.log_round(self.calls + 1, action, perturbation_status="n/a")
            return None
        name, arguments = action["tool_name"], action["arguments"]
        if name not in self.allowed:
            raise ValueError("Tool is outside this task's public tool set")
        # Validate before fault injection; otherwise malformed calls can receive valid-looking injected data.
        from jsonschema import Draft7Validator
        definition = next(tool for tool in self.definitions if tool["name"] == name)
        schema = definition.get("paradigms", {}).get("function_call", {}).get("spec", {}).get("parameters", {})
        errors = [error.message for error in Draft7Validator(schema).iter_errors(arguments)]
        if errors:
            self.validation_errors += 1
            result, status = {"error": "Validation failed", "details": errors}, "clean"
        else:
            result, status = self.engine._intercept_tool_call(name, arguments)
        self.calls += 1
        self.engine.logger.log_round(self.calls, action, result, status)
        # Only the actual output is returned to the policy; never perturbation_status.
        return result

    def score(self, token_usage=None):
        trace = self.engine.logger.to_dict(token_usage=token_usage)
        judgement = self.judge.judge(self.task, trace)
        return trace, judgement
