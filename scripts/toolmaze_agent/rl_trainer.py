"""TRL GRPO with bounded-memory reference/old-policy scoring for text trajectories."""
import torch
from torch.nn import functional as F
from trl import GRPOTrainer


def remove_frozen_aliases(module, state):
    # DeepSpeed removes frozen named_parameters() but that deduplicates tied
    # embeddings/LM-head aliases. Remove every alias as well.
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            state.pop(name, None)
    return state


@torch.no_grad()
def stateless_lora_sync_weights(generation):
    """Build merged transfer tensors without merge/unmerge roundoff in the frozen SFT base."""
    model = generation.model
    if not hasattr(model, 'peft_config'):
        raise ValueError('Stateless LoRA synchronization requires a PEFT model')
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            continue
        value = parameter.detach()
        if name.endswith('.base_layer.weight'):
            layer = model.get_submodule(name.removesuffix('.base_layer.weight'))
            delta = sum(layer.get_delta_weight(adapter).float() for adapter in layer.active_adapters)
            value = (value.float() + delta).to(parameter.dtype)
        clean = name.removeprefix('base_model.model.').replace('.base_layer', '')
        yield generation._fix_param_name_to_vllm(clean), value


class ToolMazeGRPOTrainer(GRPOTrainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        if not self.is_deepspeed_enabled or not hasattr(self.model, 'peft_config'):
            return super()._save_optimizer_and_scheduler(output_dir)
        from types import MethodType
        engine = self.model_wrapped
        original = engine.module_state_dict
        def module_state_dict(engine, *args, **kwargs):
            state = original(*args, **kwargs)
            if kwargs.get('exclude_frozen_parameters'):
                remove_frozen_aliases(engine.module, state)
            return state
        engine.module_state_dict = MethodType(module_state_dict, engine)
        try:
            return super()._save_optimizer_and_scheduler(output_dir)
        finally:
            engine.module_state_dict = original

    def _generate(self, prompts):
        if hasattr(self.rollout_func, 'has_replay') and self.rollout_func.has_replay(self.state.global_step):
            self._last_loaded_step = self.state.global_step
        return super()._generate(prompts)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
            logits_to_keep, batch_size=None, compute_entropy=False, compute_aux_loss=False, **kwargs):
        # The gradient path uses TRL's fused Liger GRPO loss. This path is only the
        # detached old/reference policy, and never materializes B x T x vocabulary.
        if torch.is_grad_enabled() or not self.use_liger_kernel:
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask,
                logits_to_keep, batch_size, compute_entropy, compute_aux_loss, **kwargs)
        if compute_aux_loss or any(v is not None for k, v in kwargs.items() if k.startswith('pixel')):
            raise ValueError('Chunked RL scoring supports dense text-only models')
        unwrapped = self.accelerator.unwrap_model(model)
        head = unwrapped.get_output_embeddings()
        all_logps, all_entropy = [], []
        forward_batch = min(batch_size or input_ids.size(0), 2)
        for start in range(0, input_ids.size(0), forward_batch):
            ids = input_ids[start:start+forward_batch]
            hidden = self._get_last_hidden_state(unwrapped, ids,
                attention_mask[start:start+forward_batch], logits_to_keep)
            target = ids[:, -logits_to_keep:]
            logps, entropies = [], []
            for offset in range(0, logits_to_keep, 128):
                logits = F.linear(hidden[:, offset:offset+128], head.weight, head.bias)
                lp = F.log_softmax(logits.float() / self.temperature, dim=-1)
                logps.append(lp.gather(-1, target[:, offset:offset+128, None]).squeeze(-1))
                if compute_entropy:
                    entropies.append(-(lp.exp() * lp).sum(-1))
            all_logps.append(torch.cat(logps, dim=1))
            if compute_entropy:
                all_entropy.append(torch.cat(entropies, dim=1))
        return torch.cat(all_logps), torch.cat(all_entropy) if compute_entropy else None, None
