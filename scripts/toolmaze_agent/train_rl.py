"""Online multi-turn GRPO using TRL's custom rollout + environment token mask."""
from __future__ import annotations

import argparse
import os

from .common import digest, load_config, project_path, provenance, read_rows, write_json
from .policy import HFBackend, load_student
from .prepare import load_task
from .rollout import episode_reward, flatten_episode, run_episode


def toolmaze_reward(prompts, completions, episode_score, **kwargs):
    return episode_score


def adapter_callback(output):
    import torch
    from transformers import TrainerCallback

    class AdapterProgress(TrainerCallback):
        def norm(self, model):
            values = [p.detach().float().square().sum() for name,p in model.named_parameters()
                      if '.lora_B.' in name]
            return torch.stack(values).sum().sqrt().item() if values else None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if state.is_world_process_zero:
                self.initial = self.norm(model)
                write_json(output/'adapter_progress.json', {'step':state.global_step,
                    'initial_lora_b_l2':self.initial,'current_lora_b_l2':self.initial,
                    'adapter_updated':False,'resumed_at_step':state.global_step})
                write_json(output/'progress.json', {'state':'running','step':state.global_step,
                    'total_steps':state.max_steps,'resumed_at_step':state.global_step})

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.is_world_process_zero:
                value = self.norm(model)
                write_json(output/'adapter_progress.json', {'step':state.global_step,
                    'initial_lora_b_l2':self.initial,'current_lora_b_l2':value,
                    'adapter_updated':value != self.initial})

        def on_save(self, args, state, control, **kwargs):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if state.is_world_process_zero:
                checkpoint = output/f'checkpoint-{state.global_step}'
                write_json(checkpoint/'checkpoint_complete.json', {'step':state.global_step,
                    'world_size':args.world_size,'optimizer_and_rng_saved':not args.save_only_model})

    return AdapterProgress()


class ToolMazeRollout:
    def __init__(self, rows, config, generation, journal):
        self.rows = {row["id"]: row for row in rows}
        self.config, self.generation, self.journal = config, generation, journal
        self.counter = 0
        self._rank_initialized = False

    def __call__(self, prompts, trainer):
        import torch
        from trl.models import unwrap_model_for_generation
        result = {"prompt_ids": [], "completion_ids": [], "env_mask": [],
                  "logprobs": None, "episode_score": []}
        if not self._rank_initialized:
            folder = self.journal / f"rank{trainer.accelerator.process_index}"
            previous = [int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()]
            self.counter = max(previous, default=-1) + 1
            self._rank_initialized = True
        # One fresh ToolMazeEnvironment is created inside each run_episode, including group siblings.
        # ZeRO-2 keeps actor weights replicated, so variable-length local rollouts need no collectives.
        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as model:
            backend = HFBackend(model, trainer.processing_class, self.generation)
            for key in prompts:
                if not isinstance(key, str) or key not in self.rows:
                    raise ValueError("RL dataset prompt must be a frozen manifest row ID")
                row = self.rows[key]
                episode = run_episode(load_task(row), backend, self.config)
                ids, completion, mask = flatten_episode(episode, trainer.processing_class)
                if len(ids) + len(completion) > self.generation["max_context_tokens"]:
                    raise ValueError("Rollout exceeds context; increase budget or reduce task scope")
                result["prompt_ids"].append(ids)
                result["completion_ids"].append(completion)
                result["env_mask"].append(mask)
                result["episode_score"].append(episode_reward(episode, self.config.get("call_penalty", 0.002)))
                rank = trainer.accelerator.process_index
                write_json(self.journal / f"rank{rank}" / f"{self.counter:08d}.json",
                           {"row": row, "step": trainer.state.global_step, **episode})
                self.counter += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/rl.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    project = load_config(config["project_config"])
    import torch
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import trl
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import set_seed
    if trl.__version__ != "1.13.0":
        raise RuntimeError("Custom rollout/env_mask API is pinned to trl==1.13.0")
    set_seed(config["seed"])
    split_path = project_path(project["splits"]) / "train.jsonl"
    rows = read_rows(split_path)
    if config.get("modes"):
        rows = [r for r in rows if r["mode"] in config["modes"]]
    if config.get("limit_tasks"):
        rows = rows[:config["limit_tasks"]]
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("RL requires nonempty train-only tasks")
    output = project_path(config["output"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Output is not empty; use --resume or a new output")
    model_path = str(project_path(config["model"]))
    model, tokenizer, frozen = load_student(model_path, config["dtype"], config.get("freeze_vision", True))
    peft_config = None
    if config.get("lora"):
        from peft import LoraConfig
        lora = dict(config['lora'])
        targets = [name for name, module in model.named_modules()
                   if name.startswith('model.language_model.layers.') and isinstance(module, torch.nn.Linear)]
        if not targets:
            raise ValueError('No language-only linear layers found for LoRA')
        peft_config = LoraConfig(task_type='CAUSAL_LM', target_modules=targets, **lora)
    training = dict(config["training"])
    training.setdefault("seed", config["seed"])
    training.setdefault("data_seed", config["seed"])
    if training.get("temperature", 1.0) != config["generation"]["temperature"]:
        raise ValueError("GRPO scoring temperature must match actual rollout temperature")
    if training.get("top_p", 1.0) != 1.0 or training.get("top_k", 0) != 0:
        raise ValueError("This rollout backend uses untruncated sampling: top_p=1 and top_k=0 required")
    if training.get("deepspeed"):
        training["deepspeed"] = str(project_path(training["deepspeed"]))
        import json
        ds = json.loads(project_path(training["deepspeed"]).read_text())
        if ds.get("zero_optimization", {}).get("stage", 0) > 2:
            raise ValueError("This rollout backend supports ZeRO-0/1/2 only; ZeRO-3 requires synchronized generation")
    # Dataset IDs route to the private environment. They are never tokenized into policy input.
    dataset = Dataset.from_list([{"prompt": row["id"]} for row in rows])
    if config.get('rollout_backend') == 'vllm_server':
        from .rl_vllm_rollout import ServerRollout
        rollout = ServerRollout(rows, project, config, output)
        from .rl_trainer import ToolMazeGRPOTrainer
        trainer_cls = ToolMazeGRPOTrainer
    else:
        rollout = ToolMazeRollout(rows, project, config['generation'], output / 'rollouts')
        trainer_cls = GRPOTrainer
    from .train_sft import training_callbacks
    trainer = trainer_cls(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        reward_funcs=toolmaze_reward,
        peft_config=peft_config, callbacks=training_callbacks(output)+([adapter_callback(output)] if peft_config else []),
        rollout_func=rollout,
        args=GRPOConfig(output_dir=str(output), **training))
    if args.resume:
        marker = project_path(args.resume)/'checkpoint_complete.json'
        import json
        saved = json.loads(marker.read_text()) if marker.exists() else {}
        if not saved.get('optimizer_and_rng_saved') or saved.get('world_size') != trainer.accelerator.num_processes:
            raise ValueError('Resume requires a completed checkpoint with optimizer and RNG state')
    if config.get('rollout_backend') == 'vllm_server':
        from types import MethodType
        from .rl_trainer import stateless_lora_sync_weights
        trainer.vllm_generation._iter_named_params = MethodType(stateless_lora_sync_weights, trainer.vllm_generation)
        # Torch 2.13 has a dynamic-shape guard failure when this tiny pointwise loss
        # recompiles. Keep fused/chunked head math and FLA kernels, without that compile.
        trainer.liger_loss.compiled = config.get('compile_rl_loss', False)
    if trainer.is_world_process_zero():
        write_json(output / "manifest.json", {**provenance(config), "project_config": project,
                   "train_split_sha256": digest(split_path), "frozen_visual_parameters": frozen,
                   "train_tasks": len(rows), "world_size": trainer.accelerator.num_processes,
                   "trainer": 'trl.GRPOTrainer' if trainer_cls is GRPOTrainer else 'ToolMazeGRPOTrainer (trl.GRPOTrainer subclass)',
                   "effective_batch_size": training['per_device_train_batch_size'] * training['gradient_accumulation_steps'] * trainer.accelerator.num_processes,
                   "generation_batch_size": trainer.args.generation_batch_size,
                   "stateless_lora_weight_sync": config.get('rollout_backend') == 'vllm_server',
                   "compile_rl_loss": config.get('compile_rl_loss', False),
                   "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                   "trainable_names": [name for name,p in trainer.model.named_parameters() if p.requires_grad],
                   "lora_target_modules": targets if peft_config else None,
                   "protocol": "json-action-v1", "environment_tokens_in_loss": False,
                   "reward": "strict official pass minus capped successful-call cost; invalid-action failure penalty"})
    result = trainer.train(resume_from_checkpoint=str(project_path(args.resume)) if args.resume else None)
    trainer.save_state()
    if config.get('save_final_model', True):
        trainer.save_model(str(output / "final"))
    if trainer.is_world_process_zero():
        if config.get('save_final_model', True):
            tokenizer.save_pretrained(output / "final")
        write_json(output / 'training_metrics.json', result.metrics)
        write_json(output / 'progress.json', {'state':'complete','step':trainer.state.global_step,
            'total_steps':trainer.state.max_steps,'metrics':result.metrics,
            'adapter':str(output/'final') if config.get('save_final_model', True) else None})


if __name__ == "__main__":
    main()
