"""Full language-model parameter SFT with assistant-only token loss."""
from __future__ import annotations

import argparse
import time
import os
from pathlib import Path

from .common import digest, load_config, project_path, provenance, read_json, read_rows, write_json
from .policy import load_student
from .protocol import token_segments


def training_callbacks(output):
    import torch
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def __init__(self):
            self.started = time.monotonic()
            self.last_logs = {}

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.last_logs = dict(logs or {})

        def on_step_end(self, args, state, control, **kwargs):
            rank = int(os.environ.get("RANK", "0"))
            value = {"state": "running", "step": state.global_step, "total_steps": state.max_steps,
                     "epoch": state.epoch, "elapsed_seconds": time.monotonic() - self.started,
                     "rank": rank, "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                     "logs": self.last_logs}
            if torch.cuda.is_available():
                value.update(gpu=torch.cuda.get_device_name(),
                             allocated_gib=torch.cuda.memory_allocated() / 2**30,
                             reserved_gib=torch.cuda.memory_reserved() / 2**30,
                             peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            write_json(output / "telemetry" / f"rank{rank}.json", value)
            if state.is_world_process_zero:
                write_json(output / "progress.json", value)

    return [Progress()]


def encode_rows(rows, tokenizer, max_length):
    encoded, dropped = [], []
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("SFT input contains a non-training row")
        ids, labels = token_segments(tokenizer, row["initial_messages"], row["segments"])
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)
        if len(ids) > max_length:
            dropped.append({"id": row["id"], "tokens": len(ids)})
            continue  # Never truncate a trajectory into an apparently successful partial path.
        if all(label == -100 for label in labels):
            raise ValueError("Training trajectory has no supervised tokens")
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    if not encoded:
        raise ValueError("No SFT rows fit the context limit")
    return encoded, dropped


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in rows])
                for key, pad in (("input_ids", self.pad), ("attention_mask", 0), ("labels", -100))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/sft.yaml")
    parser.add_argument("--resume", help="Explicit Trainer checkpoint directory")
    args = parser.parse_args()
    config = load_config(args.config)
    import torch
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(config["seed"])
    output = project_path(config["output"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Output is not empty; use --resume or a new output")
    model_path = config["model"]
    if project_path(model_path).exists():
        model_path = str(project_path(model_path))
    model, tokenizer, frozen = load_student(model_path, config["dtype"], config.get("freeze_vision", True))
    kernels = {}
    if model.config.model_type == "qwen3_5":
        from transformers.models.qwen3_5 import modeling_qwen3_5
        kernels = {name: getattr(getattr(modeling_qwen3_5, name, None), "__module__", None)
                   for name in ("chunk_gated_delta_rule", "causal_conv1d_fn")}
    source = project_path(config["data"])
    data_manifest = read_json(source.parent / "manifest.json")
    if digest(source) != data_manifest["train_sha256"]:
        raise ValueError("SFT data changed after build")
    rows, dropped = encode_rows(read_rows(source), tokenizer, config["max_length"])
    if config.get("stress_longest"):
        rows.sort(key=lambda row: len(row["input_ids"]), reverse=True)
    if config.get("max_trajectories"):
        rows = rows[:config["max_trajectories"]]
    training = dict(config["training"])
    training.setdefault("seed", config["seed"])
    training.setdefault("data_seed", config["seed"])
    if training.get("deepspeed"):
        training["deepspeed"] = str(project_path(training["deepspeed"]))
    training.setdefault("loss_type", "chunked_nll")
    training.setdefault("max_length", config["max_length"])
    training.setdefault("dataset_kwargs", {"skip_prepare_dataset": True})
    training.setdefault("packing", False)
    training.setdefault("remove_unused_columns", False)
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(output), **training),
                      train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                      data_collator=AssistantCollator(tokenizer.pad_token_id), callbacks=training_callbacks(output))
    manifest = {**provenance(config), "data_sha256": digest(source),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "frozen_visual_parameters": frozen, "accepted_trajectories": len(rows),
                "trainer": "trl.SFTTrainer", "full_language_parameter_tuning": True,
                "kernel_backends": kernels,
                "peft_enabled": hasattr(model, "peft_config"),
                "world_size": trainer.accelerator.num_processes,
                "effective_batch_size": training["per_device_train_batch_size"] * training["gradient_accumulation_steps"] * trainer.accelerator.num_processes,
                "dropped_overlength": dropped, "supervised_tokens": sum(sum(x != -100 for x in r["labels"]) for r in rows)}
    if trainer.is_world_process_zero():
        write_json(output / "manifest.json", manifest)
    result = trainer.train(resume_from_checkpoint=str(project_path(args.resume)) if args.resume else None)
    trainer.save_state()
    if trainer.is_world_process_zero():
        write_json(output / "training_metrics.json", result.metrics)
    save_final = config.get("save_final_model", True)
    linked = False
    if save_final and config.get("final_symlink_to_last_checkpoint"):
        from transformers.trainer_utils import get_last_checkpoint
        latest = get_last_checkpoint(str(output))
        if latest and latest.endswith(f"checkpoint-{trainer.state.global_step}"):
            trainer.accelerator.wait_for_everyone()
            if trainer.is_world_process_zero():
                (output / "final").symlink_to(Path(latest).name, target_is_directory=True)
            linked = True
    if save_final and not linked:
        trainer.save_model(str(output / "final"))
    if trainer.is_world_process_zero():
        if save_final:
            tokenizer.save_pretrained(output / "final")
        write_json(output / "progress.json", {"state": "complete", "step": trainer.state.global_step,
                   "total_steps": trainer.state.max_steps, "epoch": trainer.state.epoch,
                   "metrics": result.metrics, "model": str(output / "final") if save_final else None})


if __name__ == "__main__":
    main()
