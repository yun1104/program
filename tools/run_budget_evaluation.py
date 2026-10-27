#!/usr/bin/env python3
"""Evaluate all four frozen checkpoints under a separately versioned budget."""
import argparse
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import digest, load_config, read_json, write_json
from tools.run_rl import require_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--suite', default='scripts/toolmaze_agent/configs/eval_rounds25_suite.yaml')
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != len(set(gpus)):
        raise ValueError('Duplicate GPU indices')
    require_idle(gpus)
    suite = load_config(args.suite)
    report = ROOT/suite['report_output']
    report.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'.deps/sft_fast'),str(ROOT),env.get('PYTHONPATH','')])
    configs = {entry['name']:load_config(entry['config']) for entry in suite['models']}
    for config in configs.values():
        project = load_config(config['project_config'])
        if project['max_rounds'] != suite['max_rounds'] or config['trials'] != 1 or config['split'] != 'test':
            raise ValueError('Suite budget or frozen-test configuration mismatch')
    receipt = {'state':'starting','launcher_pid':os.getpid(),'gpus':gpus,'suite':suite,
               'suite_sha256':digest(ROOT/args.suite),'started_at_unix':time.time(),'models':{},
               'policy':'Fresh 25-round evaluations, no inheritance from 16-round episodes; same frozen final weights.'}
    current = None

    def heartbeat():
        completed = 0
        progress = {}
        for name,config in configs.items():
            path = ROOT/config['output']/'progress.json'
            value = read_json(path) if path.exists() else {'completed':0,'total':suite['expected_tasks']}
            progress[name] = value
            completed += value.get('completed',0)
        write_json(report/'progress.json',{'state':receipt['state'],'phase':receipt.get('phase'),
            'current_model':receipt.get('current_model'),'models':progress,
            'completed_episodes':completed,'total_episodes':suite['expected_tasks']*len(configs),
            'max_rounds':suite['max_rounds'],'updated_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')})
        write_json(report/'launch.json',receipt)

    def execute(command,log,environment=None):
        nonlocal current
        with log.open('a') as stream:
            current = subprocess.Popen(command,cwd=ROOT,env=environment or env,stdout=stream,
                                       stderr=subprocess.STDOUT,start_new_session=True)
            receipt['child_pid'] = current.pid
            while current.poll() is None:
                heartbeat()
                time.sleep(5)
            heartbeat()
            if current.returncode:
                raise RuntimeError(f'Command failed ({current.returncode}); inspect {log}')

    try:
        for entry in suite['models']:
            name=entry['name'];config=configs[name]
            receipt.update(state='running',current_model=name,phase='preparing_model')
            heartbeat()
            merged = None
            if entry.get('adapter'):
                merged = ROOT/config['model']
                if merged.exists():
                    marker = merged/'.rl_eval_temporary.json'
                    if not marker.exists() or read_json(marker)['adapter'] != str((ROOT/entry['adapter']).resolve()):
                        raise ValueError('Refusing to reuse unknown temporary model')
                    if not (ROOT/config['model_manifest']).exists():
                        raise ValueError('Temporary model has no provenance manifest')
                else:
                    if shutil.disk_usage(ROOT).free < 7*1024**3:
                        raise RuntimeError('Insufficient disk space for a temporary merged model')
                    receipt['phase']='merging_adapter'
                    execute([sys.executable,'-u','tools/merge_rl_adapter.py','--adapter',entry['adapter'],
                             '--output',str(merged),'--manifest',config['model_manifest']],
                            report/f'{name}_merge.log',dict(env,CUDA_VISIBLE_DEVICES=''))
            require_idle(gpus)
            receipt['phase']='evaluating'
            execute([sys.executable,'-u','tools/run_rl_eval_parallel.py','--gpus',args.gpus,
                     '--config',entry['config'],'--shard-dir',f'{name}_rounds25_shards','--shard-order-hash'],
                     report/f'{name}_evaluation.log')
            run=ROOT/config['output']
            summary=read_json(run/'summary.json');progress=read_json(run/'progress.json')
            manifest=read_json(run/'manifest.json')
            if (not progress.get('finished') or progress['completed'] != suite['expected_tasks']
                or manifest['project_config']['max_rounds'] != suite['max_rounds']
                or manifest.get('inherited_completed_episodes')):
                raise ValueError('Incomplete or incorrectly inherited budget evaluation')
            records=[read_json(p) for p in (run/'episodes').glob('*.json')]
            if len(records)!=suite['expected_tasks']:
                raise ValueError('Incorrect episode count')
            decisions=[sum(s['model'] for s in r.get('segments',[])) for r in records]
            if max(decisions)>suite['max_rounds']:
                raise ValueError('Evaluation exceeded its model-decision budget')
            receipt['models'][name]={'run':config['output'],'metrics':summary['breakdown']['all'],
                                    'max_model_decisions':max(decisions)}
            heartbeat()
            if merged:
                if merged.parent.resolve()!=(ROOT/'.cache/rl_eval').resolve():
                    raise ValueError('Unsafe temporary model cleanup')
                marker=read_json(merged/'.rl_eval_temporary.json')
                if marker['adapter']!=str((ROOT/entry['adapter']).resolve()):
                    raise ValueError('Temporary model ownership changed')
                shutil.rmtree(merged)
        receipt['phase']='comparing'
        execute([sys.executable,'-u','tools/compare_budget_evaluations.py','--suite',args.suite],report/'comparison.log')
        receipt.update(state='complete',phase='complete',ended_at_unix=time.time())
    except BaseException as error:
        receipt.update(state='failed',error=f'{type(error).__name__}: {error}',ended_at_unix=time.time())
        raise
    finally:
        if current is not None and current.poll() is None:
            os.killpg(current.pid,signal.SIGTERM)
            try:
                current.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(current.pid,signal.SIGKILL)
        heartbeat()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    main()
