import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.toolmaze_agent.rl_vllm_rollout import RemotePolicy, episode_sample


class Tokenizer:
    eos_token_id = 0
    def encode(self, text, **kwargs):
        return list(text.encode())
    def apply_chat_template(self, messages, **kwargs):
        return 'prompt'


class RLTests(unittest.TestCase):
    def test_checkpoint_excludes_all_frozen_tied_parameter_aliases(self):
        import torch
        from scripts.toolmaze_agent.rl_trainer import remove_frozen_aliases
        model=torch.nn.Module()
        model.embedding=torch.nn.Embedding(7,4)
        model.head=torch.nn.Linear(4,7,bias=False)
        model.head.weight=model.embedding.weight
        model.embedding.weight.requires_grad_(False)
        model.adapter=torch.nn.Parameter(torch.ones(2))
        state=remove_frozen_aliases(model,model.state_dict())
        self.assertEqual(set(state),{'adapter'})

    def test_transient_disconnect_retries_identical_request(self):
        import httpx
        payloads=[]
        class Client:
            def post(self,path,json):
                payloads.append(json.copy())
                if len(payloads)==1:
                    raise httpx.RemoteProtocolError('Server disconnected')
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{
                    'text':'ok','token_ids':[8],'logprobs':{'token_logprobs':[-.5]},'finish_reason':'stop'}]})
        cfg={'max_tokens':512,'max_context_tokens':8192,'temperature':.8}
        policy=RemotePolicy(Client(),Tokenizer(),cfg,'model',['task',0])
        with patch('scripts.toolmaze_agent.rl_vllm_rollout.time.sleep'):
            policy.generate([],[])
        self.assertEqual(payloads[0],payloads[1])
        self.assertEqual(policy.turn,1)

    def test_bad_request_is_not_retried(self):
        import httpx
        class Client:
            calls=0
            def post(self,path,json):
                self.calls+=1
                return httpx.Response(400,request=httpx.Request('POST','http://localhost/v1/completions'))
        client=Client();cfg={'max_tokens':512,'max_context_tokens':8192,'temperature':.8}
        with self.assertRaises(httpx.HTTPStatusError):
            RemotePolicy(client,Tokenizer(),cfg,'model',['task',0]).generate([],[])
        self.assertEqual(client.calls,1)

    def test_replay_recovery_rejects_changed_training_math(self):
        import tempfile
        from pathlib import Path
        from scripts.toolmaze_agent.common import load_config
        from scripts.toolmaze_agent.rl_vllm_rollout import ServerRollout
        config=load_config('scripts/toolmaze_agent/configs/rl_lora_recovery.yaml')
        project=load_config(config['project_config'])
        with tempfile.TemporaryDirectory() as directory:
            rollout=ServerRollout([],project,config,Path(directory))
            self.assertTrue(rollout.has_replay(9));self.assertFalse(rollout.has_replay(10))
            config['training']['learning_rate']*=2
            with self.assertRaises(ValueError):
                ServerRollout([],project,config,Path(directory))

    def test_fused_grpo_matches_dense_loss_and_masks_environment_gradients(self):
        import torch
        from trl.losses import FusedLinearGRPOLoss
        torch.manual_seed(2)
        fused=FusedLinearGRPOLoss(beta=.01,compiled=False,use_ref_model=True,
                                  loss_type='grpo',temperature=.8)
        for length in [9,13]:
            hidden=torch.randn(2,length,7,requires_grad=True)
            dense_hidden=hidden.detach().clone().requires_grad_(True)
            weight=torch.randn(17,7)
            target=torch.randint(0,17,(2,length))
            mask=torch.ones(2,length);mask[:,1::3]=0
            adv=torch.tensor([1.,-1.])
            lp=torch.log_softmax(torch.nn.functional.linear(dense_hidden,weight)/.8,-1).gather(-1,target[:,:,None]).squeeze(-1)
            old=lp.detach()+.01;ref=lp.detach()-.02
            actual,_=fused(_input=hidden,lin_weight=weight,selected_token_ids=target,
                attention_mask=mask,advantages=adv,old_per_token_logps=old,ref_per_token_logps=ref)
            ratio=(lp-old).exp()
            pg=-torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None])
            kl=(ref-lp).exp()-(ref-lp)-1
            expected=(((pg+.01*kl)*mask).sum(-1)/mask.sum(-1)).mean()
            actual.backward();expected.backward()
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(hidden.grad,dense_hidden.grad,atol=1e-6,rtol=1e-5)
            self.assertEqual(hidden.grad[mask==0].abs().sum().item(),0)

    def test_lora_weight_sync_preserves_frozen_base_and_merges_updates(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from scripts.toolmaze_agent.rl_trainer import stateless_lora_sync_weights
        model=get_peft_model(torch.nn.Sequential(torch.nn.Linear(4,5,bias=False)),
                            LoraConfig(r=2,lora_alpha=4,target_modules=['0']))
        layer=model.base_model.model[0]
        with torch.no_grad():
            layer.lora_B['default'].weight.normal_()
        before=layer.base_layer.weight.detach().clone()
        generation=SimpleNamespace(model=model,_fix_param_name_to_vllm=lambda n:n)
        values=dict(stateless_lora_sync_weights(generation))
        torch.testing.assert_close(values['0.weight'],before+layer.get_delta_weight('default'))
        self.assertTrue(torch.equal(before,layer.base_layer.weight))
        self.assertFalse(any('lora_' in name for name in values))

    def test_logprobs_and_environment_mask_align_without_retokenizing_actions(self):
        episode = {'initial_messages': [{'role':'user','content':'x'}],
            'segments': [{'text':'action','model':True,'token_ids':[11,12], 'sampling_logprobs':[-1.,-2.]},
                         {'text':'obs','model':False},
                         {'text':'done','model':True,'token_ids':[13], 'sampling_logprobs':[-3.]}],
            'strict_pass':True,'tool_calls':1,'invalid_actions':0,'validation_errors':0}
        value = episode_sample(episode, Tokenizer(), 0.002)
        self.assertEqual(value['completion_ids'], [11,12,111,98,115,13,0])
        self.assertEqual(value['env_mask'], [1,1,0,0,0,1,0])
        self.assertEqual(value['logprobs'], [-1.,-2.,0.,0.,0.,-3.,0.])
        self.assertAlmostEqual(value['episode_score'], .998)

    def test_group_siblings_use_different_sampler_seeds(self):
        payloads=[]
        class Client:
            def post(self, path, json):
                payloads.append(json)
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{
                    'text':'ok','token_ids':[8], 'logprobs':{'token_logprobs':[-.5]},'finish_reason':'stop'}]})
        cfg={'max_tokens':512,'max_context_tokens':8192,'temperature':.8}
        for index in [0,1]:
            RemotePolicy(Client(),Tokenizer(),cfg,'model',['task',index]).generate([],[])
        self.assertNotEqual(payloads[0]['seed'],payloads[1]['seed'])
        self.assertEqual(payloads[0]['prompt'],list(b'prompt'))

    def test_chunked_old_policy_matches_dense_logprobs_and_entropy(self):
        import torch
        from scripts.toolmaze_agent.rl_trainer import ToolMazeGRPOTrainer
        torch.manual_seed(1)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed=torch.nn.Embedding(17,7);self.head=torch.nn.Linear(7,17)
            def get_output_embeddings(self):
                return self.head
        model=Model()
        trainer=object.__new__(ToolMazeGRPOTrainer)
        trainer.accelerator=SimpleNamespace(unwrap_model=lambda m:m)
        trainer.use_liger_kernel=True;trainer.temperature=.8
        trainer._get_last_hidden_state=lambda m,ids,mask,keep:m.embed(ids)[:,:-1][:,-keep:]
        ids=torch.randint(0,17,(3,291));keep=283
        with torch.no_grad():
            actual,entropy,_=trainer._get_per_token_logps_and_entropies(model,ids,torch.ones_like(ids),keep,compute_entropy=True)
            lp=torch.log_softmax(model.head(model.embed(ids)[:,:-1][:,-keep:])/.8,-1)
            expected=lp.gather(-1,ids[:,-keep:,None]).squeeze(-1)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(entropy,-(lp.exp()*lp).sum(-1))


if __name__=='__main__':
    unittest.main()
