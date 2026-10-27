#!/usr/bin/env python3
"""Tiny random-model SFT + real ToolMaze rollout/GRPO integration smoke test.

No API calls, pretrained weight downloads, or benchmark performance claims.
GRPO uses alternating synthetic rewards solely to verify nonzero updates.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="runs/smoke")
    parser.add_argument("--architecture", choices=["gpt2", "qwen3.5"], default="gpt2")
    args = parser.parse_args()
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from scripts.toolmaze_agent.common import load_config, project_path, read_rows, write_json
    from scripts.toolmaze_agent.train_sft import AssistantCollator, encode_rows
    from scripts.toolmaze_agent.train_rl import ToolMazeRollout, toolmaze_reward
    from scripts.toolmaze_agent.protocol import encode_action
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    set_seed(20261026)
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError("Choose a fresh smoke output directory")
    tokenizer_impl = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_impl.decoder = decoders.ByteLevel()
    tokenizer_impl.train_from_iterator(["system user assistant flight weather tool_call final_answer arguments content"],
                                      trainers.BpeTrainer(vocab_size=280, special_tokens=["<pad>", "<eos>", "<unk>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, pad_token="<pad>",
                                       eos_token="<eos>", unk_token="<unk>", model_max_length=16384)
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=16384,
                                      n_embd=16, n_layer=1, n_head=2,
                                      bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
    model.config.use_cache = False
    if args.architecture == "qwen3.5":
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config={"vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                         "head_dim": 16, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                         "linear_num_key_heads": 1, "linear_num_value_heads": 2,
                         "layer_types": ["linear_attention", "full_attention"],
                         "max_position_embeddings": 16384, "eos_token_id": tokenizer.eos_token_id,
                         "pad_token_id": tokenizer.pad_token_id,
                         "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                             "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                           "out_hidden_size": 32, "patch_size": 2, "spatial_merge_size": 1},
            image_token_id=len(tokenizer) - 1, video_token_id=len(tokenizer) - 2,
            vision_start_token_id=len(tokenizer) - 3, vision_end_token_id=len(tokenizer) - 4))
        for name, parameter in model.named_parameters():
            if "visual" in name.split("."):
                parameter.requires_grad_(False)
        model.config.use_cache = False
    if args.device == "cuda":
        model.to("cuda")
    before = model.get_input_embeddings().weight.detach().clone()
    fixture = [{"id": "smoke", "split": "train", "initial_messages": [
        {"role": "system", "content": "Return one JSON action."}, {"role": "user", "content": "Say done."}],
        "segments": [{"text": encode_action({"type": "final_answer", "content": "done"}), "model": True}]}]
    rows, _ = encode_rows(fixture, tokenizer, 512)
    trainer = Trainer(model=model, processing_class=tokenizer,
                      args=TrainingArguments(output_dir=str(output / "sft"), max_steps=1,
                          per_device_train_batch_size=1, learning_rate=.001, report_to="none",
                          save_strategy="no", use_cpu=args.device == "cpu", dataloader_pin_memory=False),
                      train_dataset=Dataset.from_list(rows), data_collator=AssistantCollator(tokenizer.pad_token_id))
    trainer.train()
    if torch.equal(before.to(model.device), model.get_input_embeddings().weight):
        raise RuntimeError("SFT did not update parameters")
    trainer.save_model(str(output / "tiny_sft"))
    tokenizer.save_pretrained(output / "tiny_sft")
    before_rl = model.get_input_embeddings().weight.detach().clone()
    project = load_config("scripts/toolmaze_agent/configs/project.yaml")
    project["max_rounds"] = 1
    task_rows = read_rows(project_path(project["splits"]) / "train.jsonl")[:1]
    production_rollout = ToolMazeRollout(task_rows, project,
                         {"max_tokens": 8, "max_context_tokens": 16384, "temperature": 1.0}, output / "rollouts")
    mask_counts = []

    def smoke_rollout(prompts, grpo):
        result = production_rollout(prompts, grpo)
        # Synthetic rewards make this a gradient-path check, not a task-success experiment.
        result["episode_score"] = [float(i % 2) for i in range(len(prompts))]
        for ids, mask in zip(result["completion_ids"], result["env_mask"]):
            if len(ids) != len(mask) or not any(mask) or all(mask):
                raise RuntimeError("Missing model/observation mask")
            mask_counts.append({"model": sum(mask), "environment": len(mask) - sum(mask)})
        return result

    grpo = GRPOTrainer(model=model, processing_class=tokenizer,
                       args=GRPOConfig(output_dir=str(output / "grpo"), max_steps=1,
                           per_device_train_batch_size=2, gradient_accumulation_steps=1,
                           num_generations=2, max_completion_length=16384, beta=0., learning_rate=.001,
                           temperature=1.0, top_p=1.0, top_k=0, use_cpu=args.device == "cpu",
                           gradient_checkpointing=False, bf16=False, fp16=False,
                           report_to="none", save_strategy="no", mask_truncated_completions=False,
                           dataloader_pin_memory=False, remove_unused_columns=False),
                       train_dataset=Dataset.from_list([{"prompt": task_rows[0]["id"]}]),
                       reward_funcs=toolmaze_reward, rollout_func=smoke_rollout)
    grpo.train()
    if torch.equal(before_rl.to(model.device), model.get_input_embeddings().weight):
        raise RuntimeError("GRPO did not update parameters")
    write_json(output / "summary.json", {"device": args.device, "architecture": args.architecture,
               "sft_updated": True, "grpo_updated": True,
               "mask_counts": mask_counts, "benchmark_result": False,
               "note": "Tiny random model + actual ToolMaze runtime; alternating synthetic RL reward"})
    print(f"Smoke passed -> {output / 'summary.json'}")


if __name__ == "__main__":
    main()
