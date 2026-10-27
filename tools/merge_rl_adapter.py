#!/usr/bin/env python3
"""Merge an RL adapter into its frozen SFT base on CPU, with provenance."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import digest, project_path, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    adapter, output = project_path(args.adapter), project_path(args.output)
    base = project_path('runs/sft_full_trl_20261026/final').resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite a merged model')
    metadata = read_json(adapter/'adapter_config.json')
    if Path(metadata['base_model_name_or_path']).resolve() != base:
        raise ValueError('Adapter was not trained on the expected SFT base')
    import torch
    from peft import PeftModel
    from scripts.toolmaze_agent.policy import load_student
    torch.set_num_threads(8)
    print(f'Loading frozen SFT base on CPU: {base}', flush=True)
    model, tokenizer, _ = load_student(str(base), 'bfloat16', True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    layer = next(layer for layer in model.modules() if hasattr(layer, 'get_delta_weight'))
    before = layer.base_layer.weight.detach().clone()
    delta = layer.get_delta_weight(layer.active_adapters[0]).detach().clone()
    if not torch.isfinite(delta).all() or not delta.ne(0).any():
        raise ValueError('Adapter has no finite nonzero update')
    print('Merging LoRA on CPU with the training sampler formula', flush=True)
    # PEFT safe_merge casts the delta to BF16 before addition. Training's
    # stateless sampler synchronization adds in FP32 and rounds only once.
    # Use the same arithmetic here, then strip adapters without merging again.
    from peft.tuners.lora import LoraLayer
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            base_layer = module.get_base_layer()
            update = sum(module.get_delta_weight(name).float() for name in module.active_adapters)
            value = (base_layer.weight.float() + update).to(base_layer.weight.dtype)
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite merged projection')
            base_layer.weight.copy_(value)
            count += 1
    if count == 0:
        raise ValueError('No LoRA layers merged')
    merged = model.unload()
    torch.testing.assert_close(layer.base_layer.weight, (before + delta).to(before.dtype), rtol=0, atol=0)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size='6GB')
    tokenizer.save_pretrained(output)
    write_json(output/'.rl_eval_temporary.json', {'adapter': str(adapter.resolve()), 'base': str(base)})
    manifest = {'base_model': str(base), 'adapter': str(adapter.resolve()),
        'adapter_sha256': digest(adapter/'adapter_model.safetensors'),
        'base_model_manifest': 'artifacts/qwen35_2b_sft_model.json',
        'merge': 'Training sampler formula: (base.float() + delta.float()).to(BF16); unload without a second merge',
        'merged_lora_layers': count,
        'temporary_merged_path': str(output.resolve()),
        'weights': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                    for p in sorted(output.glob('*.safetensors'))},
        'metadata_sha256': {p.name: digest(p) for p in sorted(output.iterdir())
                            if p.suffix in ('.json', '.jinja') and not p.name.startswith('.')}}
    write_json(project_path(args.manifest), manifest)
    print(f'Merged model ready: {output}', flush=True)


if __name__ == '__main__':
    main()
