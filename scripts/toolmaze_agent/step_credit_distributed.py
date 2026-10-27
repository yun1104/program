"""Global StepCredit credit assignment before slicing trajectories back to ranks."""
import torch

from .common import write_json
from .step_credit import assign_advantages
from .step_credit_trainer import ToolMazeStepCreditTrainer
from .rl_trainer import ToolMazeGRPOTrainer


def local_advantages(samples, counts, rank, group_size, gamma, omega):
    if sum(counts) != len(samples) or not 0 <= rank < len(counts):
        raise ValueError('Invalid gathered rollout counts')
    vectors, actions, stats = assign_advantages(samples, group_size, gamma, omega)
    start = sum(counts[:rank])
    return vectors[start:start+counts[rank]], actions, stats


class DistributedStepCreditTrainer(ToolMazeStepCreditTrainer):
    def _generate_and_score_completions(self, inputs):
        from accelerate.utils import gather_object
        # Bypass the smoke subclass's local-only advantage assignment.
        scored = ToolMazeGRPOTrainer._generate_and_score_completions(self, inputs)
        local = self.rollout_func.last_samples
        samples = gather_object(local)
        counts = gather_object([len(local)])
        vectors, actions, stats = local_advantages(samples, counts,
            self.accelerator.process_index, self.num_generations,
            self.step_credit_config.get('gamma', .95), self.step_credit_config.get('omega', 1.0))
        advantages = torch.zeros_like(scored['completion_ids'], dtype=torch.float32)
        for i, values in enumerate(vectors):
            if scored['completion_ids'][i, :len(values)].tolist() != local[i]['completion_ids']:
                raise ValueError('TRL changed distributed trajectory order or token boundaries')
            advantages[i, :len(values)] = torch.tensor(values, device=advantages.device)
        scored['advantages'] = advantages
        if self.accelerator.is_main_process:
            write_json(self.rollout_func.output/'advantages'/f'{self.state.global_step:04d}.json',
                {'step': self.state.global_step, 'statistics': stats, 'actions': actions,
                 'rank_sample_counts': counts, 'global_grouping': True})
        for key in ('shared_noninitial_actions', 'nonzero_noninitial_step_advantages'):
            self._metrics['train']['step_credit/'+key].append(float(stats[key]))
        return scored
