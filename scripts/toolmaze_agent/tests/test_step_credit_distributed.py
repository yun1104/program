"""Groups split across ranks must produce the same credit as one global batch."""
import unittest
from .test_step_credit import sample
from scripts.toolmaze_agent.step_credit import assign_advantages
from scripts.toolmaze_agent.step_credit_distributed import local_advantages
from scripts.toolmaze_agent.step_credit_server_rollout import reconstruct_steps
from scripts.toolmaze_agent.common import fingerprint
from scripts.toolmaze_agent.protocol import token_segments


class DistributedStepCreditTests(unittest.TestCase):
    def test_group_crosses_rank_boundary(self):
        samples = [sample(v) for v in (1, 0, .5, -.1)]
        expected, _, _ = assign_advantages(samples, 4)
        left, _, _ = local_advantages(samples, [2, 2], 0, 4, .95, 1.)
        right, _, _ = local_advantages(samples, [2, 2], 1, 4, .95, 1.)
        self.assertEqual(left+right, expected)
        self.assertNotEqual(left[0], assign_advantages(samples[:2], 2)[0][0])

    def test_anchor_reconstruction_uses_actual_token_ids(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())
            def apply_chat_template(self, *args, **kwargs):
                return 'prompt'
        tokenizer = Tokenizer()
        episode = {'initial_messages': [], 'segments': [
            {'model': True, 'text': 'different text', 'token_ids': [901, 902]},
            {'model': False, 'text': 'obs'},
            {'model': True, 'text': 'finish', 'token_ids': [903]}]}
        prefix, _ = token_segments(tokenizer, [], [])
        steps = reconstruct_steps(episode, tokenizer, 'task')
        self.assertEqual(steps[0], {'anchor': fingerprint(['task', prefix]), 'start': 0, 'end': 2})
        self.assertEqual(steps[1], {'anchor': fingerprint(['task', prefix+[901,902]+list(b'obs')]),
                                   'start': 5, 'end': 6})


if __name__ == '__main__':
    unittest.main()
