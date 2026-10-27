"""Credit assignment, token mask and gradient regression tests."""
import copy
import math
import unittest
import torch
from scripts.toolmaze_agent.gigpo import assign_advantages, normalized, policy_token_loss


def sample(reward, anchor='shared', row='train-task'):
    return {'row_id': row, 'episode_score': reward, 'completion_ids': [1, 2, 3, 4, 5],
            'env_mask': [1, 1, 0, 1, 0], 'action_steps': [
                {'anchor': 'initial', 'start': 0, 'end': 2},
                {'anchor': anchor, 'start': 3, 'end': 4}]}


def test_episode_and_step_formula_and_masks():
    samples = [sample(1), sample(0)]
    vectors, rows, stats = assign_advantages(samples, 2, gamma=.9, omega=.5)
    episode = normalized([1, 0])[0]
    assert math.isclose(vectors[0][0], episode + .5 * normalized([.9, 0])[0])
    assert math.isclose(vectors[0][3], 1.5 * episode)
    assert vectors[0][2] == vectors[0][4] == 0
    assert stats['nonzero_noninitial_step_advantages'] == 2


def test_singleton_fallback_and_omega_zero():
    samples = [sample(1, 'a'), sample(0, 'b')]
    vectors, _, stats = assign_advantages(samples, 2)
    assert math.isclose(vectors[0][3], normalized([1, 0])[0])
    assert stats['shared_noninitial_actions'] == 0
    vectors, _, _ = assign_advantages(samples, 2, omega=0)
    assert vectors[0][0] == vectors[0][3]


def test_groups_never_cross_tasks_or_rollout_groups():
    samples = [sample(1), sample(0), sample(0), sample(0)]
    vectors, _, _ = assign_advantages(samples, 2)
    assert vectors[2] == [0] * 5
    wrong = copy.deepcopy(samples[:2])
    wrong[1]['row_id'] = 'another-task'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'same training task'):
        assign_advantages(wrong, 2)


def test_reject_environment_credit_and_missing_model_spans():
    samples = [sample(1), sample(0)]
    samples[0]['action_steps'][0]['end'] = 3
    with unittest.TestCase().assertRaisesRegex(ValueError, 'environment'):
        assign_advantages(samples, 2)
    samples = [sample(1), sample(0)]
    samples[0]['action_steps'][0]['end'] = 1
    with unittest.TestCase().assertRaisesRegex(ValueError, 'cover precisely'):
        assign_advantages(samples, 2)


def test_token_loss_different_turn_gradients_and_environment_mask():
    lp = torch.tensor([[-.5, -.7, -.3]], requires_grad=True)
    old = lp.detach().clone()
    advantage = torch.tensor([[1., -1., 0.]])
    losses = policy_token_loss(lp, old, old, advantage, 0, .2, .2)
    mask = torch.tensor([[1., 1., 0.]])
    (losses * mask).sum().backward()
    assert lp.grad[0, 0] < 0 and lp.grad[0, 1] > 0
    assert lp.grad[0, 2] == 0


def test_checkpointed_head_matches_dense_loss_and_hidden_gradients():
    from collections import defaultdict
    from types import SimpleNamespace
    from scripts.toolmaze_agent.gigpo_trainer import ToolMazeGiGPOTrainer
    torch.manual_seed(3)
    weight = torch.randn(19, 7)
    h1 = torch.randn(2, 5, 7, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_()
    ids = torch.randint(19, (2, 5))
    advantages = torch.randn(2, 5)
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 1, 0]])
    def lp(h, target):
        return (h @ weight.T).log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    dense = lp(h1, ids)
    old, ref = dense.detach() - .05, dense.detach() + .02
    loss1 = ((policy_token_loss(dense, old, ref, advantages, .01, .2, .2) * mask).sum(-1)/mask.sum(-1)).mean()
    trainer = SimpleNamespace(_get_last_hidden_state=lambda *args: h2, temperature=1.,
        gigpo_config={'head_chunk_tokens': 2}, beta=.01, epsilon_low=.2, epsilon_high=.2,
        _metrics={'train': defaultdict(list)}, current_gradient_accumulation_steps=1)
    head = SimpleNamespace(weight=weight, bias=None)
    model = SimpleNamespace(get_output_embeddings=lambda: head)
    inputs = {'prompt_ids': torch.zeros(2, 1, dtype=torch.long),
        'prompt_mask': torch.ones(2, 1, dtype=torch.long),
        'completion_ids': ids, 'completion_mask': torch.ones_like(mask), 'tool_mask': mask,
        'old_per_token_logps': old, 'ref_per_token_logps': ref, 'advantages': advantages}
    loss2 = ToolMazeGiGPOTrainer.compute_liger_loss(trainer, model, inputs)
    loss1.backward()
    loss2.backward()
    torch.testing.assert_close(loss1, loss2)
    torch.testing.assert_close(h1.grad, h2.grad)


class GiGPOTests(unittest.TestCase):
    pass


for name, function in list(globals().items()):
    if name.startswith("test_") and callable(function):
        setattr(GiGPOTests, name, lambda self, function=function: function())
del name, function


if __name__ == "__main__":
    unittest.main()
