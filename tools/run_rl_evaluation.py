#!/usr/bin/env python3
"""Evaluate final GRPO/StepCredit adapters on the frozen test set, without selection."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import load_config, read_json, write_json
from tools.run_rl import require_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    gpus = [int(v) for v in args.gpus.split(',')]
    if len(gpus) != len(set(gpus)):
        raise ValueError('Duplicate GPU selection')
    require_idle(gpus)
    report = ROOT/'reports/rl_eval/grpo_vs_step_credit_20261027'
    report.mkdir(parents=True, exist_ok=True)
    receipt = {'state': 'starting', 'gpus': gpus, 'launcher_pid': os.getpid(),
               'started_at_unix': time.time(), 'models': {}}
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'.deps/sft_fast'), str(ROOT), env.get('PYTHONPATH','')])
    current = None
    def execute(command, log, extra_env=None):
        nonlocal current
        with log.open('a') as stream:
            current = subprocess.Popen(command, cwd=ROOT, env=extra_env or env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            receipt['child_pid'] = current.pid
            write_json(report/'launch.json', receipt)
            code = current.wait()
            if code:
                raise RuntimeError(f'Command failed ({code}); inspect {log}')
    try:
        for name, adapter in [('grpo','runs/grpo_lora_recovery_20261026/final'),
                              ('step_credit','runs/step_credit_full_20261027/final')]:
            merged = ROOT/f'.cache/rl_eval/{name}_final_20261027'
            manifest = f'artifacts/qwen35_2b_{name}_model_20261027.json'
            config_path = f'scripts/toolmaze_agent/configs/{name}_eval_vllm_20261027.yaml'
            config = load_config(config_path)
            receipt.update(state='merging', current_model=name)
            write_json(report/'launch.json',receipt)
            if not merged.exists():
                if shutil.disk_usage(ROOT).free < 7 * 1024**3:
                    raise RuntimeError('Insufficient free disk space for a temporary merged model')
                execute([sys.executable,'-u','tools/merge_rl_adapter.py','--adapter',adapter,
                         '--output',str(merged),'--manifest',manifest], report/f'{name}_merge.log',
                         dict(env, CUDA_VISIBLE_DEVICES=''))
            elif not (merged/'.rl_eval_temporary.json').exists() or not (ROOT/manifest).exists():
                raise RuntimeError('Incomplete or unknown merged model; refusing to reuse it')
            require_idle(gpus)
            receipt['state'] = 'evaluating'
            write_json(report/'launch.json',receipt)
            execute([sys.executable,'-u','tools/run_rl_eval_parallel.py','--gpus',args.gpus,
                     '--config',config_path,'--shard-dir',name+'_eval_shards','--shard-order-hash'],
                     report/f'{name}_evaluation.log')
            summary = read_json(ROOT/config['output']/'summary.json')
            progress = read_json(ROOT/config['output']/'progress.json')
            if not progress.get('finished') or progress['completed'] != 200:
                raise RuntimeError('Evaluation did not cover the full 200-task test set')
            receipt['models'][name] = {'run': config['output'], 'summary': summary['breakdown']['all']}
            write_json(report/'launch.json',receipt)
            # Only remove this launcher's explicitly marked, regenerable merged
            # copy after all episodes/metadata are saved. Never remove adapters.
            if merged.parent.resolve() != (ROOT/'.cache/rl_eval').resolve():
                raise ValueError('Unsafe temporary cleanup path')
            marker = read_json(merged/'.rl_eval_temporary.json')
            if marker['adapter'] != str((ROOT/adapter).resolve()):
                raise ValueError('Temporary merge belongs to another adapter')
            shutil.rmtree(merged)
        receipt['state'] = 'comparing'
        write_json(report/'launch.json',receipt)
        execute([sys.executable,'-u','tools/compare_rl_evaluations.py'],report/'comparison.log')
        receipt.update(state='complete',ended_at_unix=time.time())
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
        write_json(report/'launch.json',receipt)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    main()
