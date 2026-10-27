"""Full train-split distributed GiGPO from the frozen SFT checkpoint."""
import argparse
import os
from types import MethodType

from .common import digest, load_config, project_path, read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = load_config(args.config)
    project = load_config(config['project_config'])
    output = project_path(config['output'])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh output directory or --resume')
    import torch
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOConfig
    from .policy import load_student
    from .train_rl import adapter_callback, toolmaze_reward
    from .train_sft import training_callbacks
    from .rl_trainer import stateless_lora_sync_weights
    from .gigpo_server_rollout import GiGPOServerRollout
    from .gigpo_distributed import DistributedGiGPOTrainer
    if trl.__version__ != '1.13.0':
        raise RuntimeError('GiGPO integration is verified against TRL 1.13.0')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    set_seed(config['seed'])
    split = project_path(project['splits'])/'train.jsonl'
    available = {r['id']: r for r in read_rows(split)}
    rows = list(available.values())
    if len(rows) != config['expected_train_tasks']:
        raise ValueError('Full training split size changed')
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Training tasks must belong to the frozen train split')
    model, tokenizer, frozen = load_student(str(project_path(config['model'])), 'bfloat16', True)
    targets = [name for name, layer in model.named_modules()
               if name.startswith('model.language_model.layers.') and isinstance(layer, torch.nn.Linear)]
    if not targets:
        raise ValueError('No language LoRA targets')
    lora = LoraConfig(task_type='CAUSAL_LM', target_modules=targets, **config['lora'])
    training = dict(config['training'])
    training.update(seed=config['seed'], data_seed=config['seed'])
    training['deepspeed'] = str(project_path(training['deepspeed']))
    if training['temperature'] != config['generation']['temperature']:
        raise ValueError('Generation and scoring temperatures must match')
    rollout = GiGPOServerRollout(rows, project, config, output)
    trainer = DistributedGiGPOTrainer(model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list([{'prompt': r['id']} for r in rows]),
        reward_funcs=toolmaze_reward, peft_config=lora, rollout_func=rollout,
        callbacks=training_callbacks(output)+[adapter_callback(output)],
        args=GRPOConfig(output_dir=str(output), **training), gigpo_config=config['gigpo'])
    if trainer.accelerator.num_processes != config['expected_training_gpus']:
        raise ValueError('Training world size does not match the full-run configuration')
    expected_trajectories = len(rows) * trainer.num_generations
    if trainer.args.max_steps * trainer.args.generation_batch_size != expected_trajectories:
        raise ValueError('Full run must cover exactly one task pass at the configured group size')
    trainer.vllm_generation._iter_named_params = MethodType(stateless_lora_sync_weights, trainer.vllm_generation)
    trainer.liger_loss.compiled = False  # custom chunked token loss, not Liger's scalar-advantage kernel
    if args.resume:
        import json
        saved = json.loads((project_path(args.resume)/'checkpoint_complete.json').read_text())
        if not saved.get('optimizer_and_rng_saved') or saved.get('world_size') != trainer.accelerator.num_processes:
            raise ValueError('Resume requires a completed native checkpoint with the same world size')
    if trainer.is_world_process_zero():
        write_json(output/'manifest.json', {
            'config': config, 'train_split_sha256': digest(split), 'train_rows': rows,
            'base_model': str(project_path(config['model']).resolve()),
            'algorithm': 'GiGPO: episode normalized return + omega * anchor-normalized discounted return',
            'anchor': 'exact full policy token prefix, same task and rollout group',
            'process_reward_model': False, 'extra_rollouts': False, 'environment_tokens_in_loss': False,
            'training_stack': 'TRL + DeepSpeed ZeRO-2 + PEFT; dedicated vLLM server',
            'loss': 'tokenwise clipped surrogate + KL; checkpointed frozen output head',
            'smoke_only': False, 'world_size': trainer.accelerator.num_processes, 'train_tasks': len(rows), 'effective_batch_size': trainer.args.generation_batch_size, 'frozen_visual_parameters': frozen,
            'trainable_parameters': sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            'lora_target_modules': targets})
    result = trainer.train(resume_from_checkpoint=str(project_path(args.resume)) if args.resume else None)
    trainer.save_state()
    trainer.save_model(str(output/'final'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output/'final')
        write_json(output/'training_metrics.json', result.metrics)
        write_json(output/'progress.json', {'state': 'complete', 'step': trainer.state.global_step,
            'total_steps': trainer.state.max_steps, 'metrics': result.metrics})


if __name__ == '__main__':
    main()
