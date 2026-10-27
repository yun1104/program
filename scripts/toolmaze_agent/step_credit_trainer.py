"""Independent TRL StepCredit subclass with bounded-memory tokenwise loss.

The installed Liger GRPO kernel accepts only one advantage per sequence.
Using it unchanged would silently erase StepCredit's action-level distinction.
This subclass instead checkpoints the frozen LM head in small token chunks.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import write_json
from .step_credit import assign_advantages, policy_token_loss
from .rl_trainer import ToolMazeGRPOTrainer


class ToolMazeStepCreditTrainer(ToolMazeGRPOTrainer):
    def __init__(self, *args, step_credit_config=None, **kwargs):
        self.step_credit_config = step_credit_config or {}
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        scored = super()._generate_and_score_completions(inputs)
        samples = self.rollout_func.last_samples
        vectors, rows, stats = assign_advantages(samples, self.num_generations,
            self.step_credit_config.get('gamma', .95), self.step_credit_config.get('omega', 1.0))
        advantages = torch.zeros_like(scored['completion_ids'], dtype=torch.float32)
        for i, values in enumerate(vectors):
            if scored['completion_ids'][i, :len(values)].tolist() != samples[i]['completion_ids']:
                raise ValueError('TRL changed completion order or token boundaries')
            advantages[i, :len(values)] = torch.tensor(values, device=advantages.device)
        scored['advantages'] = advantages
        write_json(self.rollout_func.output/'advantages'/f'{self.state.global_step:04d}.json',
                   {'step': self.state.global_step, 'statistics': stats, 'actions': rows})
        for key in ('shared_noninitial_actions', 'nonzero_noninitial_step_advantages'):
            self._metrics['train']['step_credit/'+key].append(float(stats[key]))
        return scored

    def compute_liger_loss(self, unwrapped_model, inputs):
        """TRL's fused-loss dispatch, replaced here by a token-advantage loss."""
        ids = torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1)
        attention = torch.cat([inputs['prompt_mask'], inputs['completion_mask']], dim=1)
        length = inputs['completion_ids'].size(1)
        hidden = self._get_last_hidden_state(unwrapped_model, ids, attention, length)
        head = unwrapped_model.get_output_embeddings()
        if head.weight.requires_grad or (head.bias is not None and head.bias.requires_grad):
            raise ValueError('StepCredit chunked head requires a frozen output embedding')
        mask = inputs['completion_mask'] * inputs['tool_mask']
        total = hidden.sum() * 0
        chunk_size = self.step_credit_config.get('head_chunk_tokens', 64)
        kl_values = []
        for i in range(hidden.size(0)):
            positions = mask[i].nonzero(as_tuple=True)[0]
            denominator = mask[i].sum().clamp_min(1)
            for positions_chunk in positions.split(chunk_size):
                target = inputs['completion_ids'][i, positions_chunk]
                def logps_for_chunk(h, target):
                    logits = F.linear(h, head.weight, head.bias).float() / self.temperature
                    return logits.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
                lp = checkpoint(logps_for_chunk, hidden[i, positions_chunk], target, use_reentrant=False)
                old = inputs['old_per_token_logps'][i, positions_chunk].detach()
                ref = inputs['ref_per_token_logps'][i, positions_chunk].detach() if self.beta else None
                correction = inputs.get('importance_sampling_ratio')
                if correction is not None:
                    correction = correction[i, positions_chunk].detach()
                token_loss = policy_token_loss(lp, old, ref, inputs['advantages'][i, positions_chunk],
                    self.beta, self.epsilon_low, self.epsilon_high, correction)
                total = total + token_loss.sum() / denominator / hidden.size(0)
                if self.beta:
                    delta = ref - lp.detach()
                    kl_values.append((delta.exp() - delta - 1).sum())
        if kl_values:
            self._metrics['train']['kl'].append((torch.stack(kl_values).sum()/mask.sum()).item())
        return total / self.current_gradient_accumulation_steps
