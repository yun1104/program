"""Model decision limits include final_answer and failed action attempts."""
import unittest
from unittest.mock import patch
from scripts.toolmaze_agent.rollout import run_episode


class Environment:
    def __init__(self,*args):
        self.query='test';self.definitions=[];self.allowed={'ping'}
        self.finished=False;self.validation_errors=0;self.calls=0
    def step(self,action):
        if action['type']=='final_answer':self.finished=True
        else:self.calls+=1
        return {'ok':True}
    def score(self,usage):return {'messages':[]},{'pass':True}


class RoundBudgetTests(unittest.TestCase):
    def test_final_answer_on_round_25_is_allowed(self):
        class Policy:
            calls=0
            def generate(self,*args):
                self.calls+=1
                return {'text':'{"type":"final_answer","content":"done"}</action>' if self.calls==25
                    else '{"type":"tool_call","tool_name":"ping","arguments":{}}</action>'}
        with patch('scripts.toolmaze_agent.rollout.ToolMazeEnvironment',Environment):
            backend=Policy();episode=run_episode({},backend,{'max_rounds':25})
        self.assertEqual(backend.calls,25);self.assertEqual(episode['tool_calls'],24)
        self.assertTrue(episode['strict_pass']);self.assertEqual(episode['termination'],'final_answer')

    def test_same_policy_is_cut_off_at_16_under_old_budget(self):
        class Policy:
            calls=0
            def generate(self,*args):
                self.calls+=1
                return {'text':'{"type":"tool_call","tool_name":"ping","arguments":{}}</action>'}
        with patch('scripts.toolmaze_agent.rollout.ToolMazeEnvironment',Environment):
            backend=Policy();episode=run_episode({},backend,{'max_rounds':16})
        self.assertEqual(backend.calls,16);self.assertEqual(episode['termination'],'max_rounds')
        self.assertFalse(episode['strict_pass'])

    def test_invalid_actions_also_consume_rounds(self):
        class Policy:
            calls=0
            def generate(self,*args):self.calls+=1;return {'text':'invalid'}
        with patch('scripts.toolmaze_agent.rollout.ToolMazeEnvironment',Environment):
            backend=Policy();episode=run_episode({},backend,{'max_rounds':25})
        self.assertEqual(backend.calls,25);self.assertEqual(episode['invalid_actions'],25)
        self.assertEqual(episode['tool_calls'],0)


if __name__=='__main__':unittest.main()
