#!/usr/bin/env python3
"""Launch isolated single-GPU GiGPO; never reuse or stop the existing sampler."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import digest, load_config, write_json
from tools.run_rl import require_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--config', default='scripts/toolmaze_agent/configs/gigpo_smoke.yaml')
    parser.add_argument('--resume')
    args = parser.parse_args()
    require_idle([args.gpu])
    config = load_config(args.config)
    report = ROOT/'reports/gigpo'/Path(config['output']).name
    report.mkdir(parents=True, exist_ok=True)
    protected = list((ROOT/'scripts/toolmaze_agent').glob('*.py')) + [ROOT/'tools/run_rl.py']
    protected = [p for p in protected if 'gigpo' not in p.name]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_CACHE_ROOT=str(ROOT/'.cache/vllm_gigpo'),
        TRITON_CACHE_DIR=str(ROOT/'.cache/triton_sft'),
        TORCH_EXTENSIONS_DIR=str(ROOT/'.cache/torch_extensions'))
    # vLLM sleep mode uses its own CUDA memory pool; expandable_segments conflicts.
    env.pop('PYTORCH_ALLOC_CONF', None)
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'.deps/rl'), str(ROOT/'.deps/sft_fast'), str(ROOT), env.get('PYTHONPATH', '')])
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
    toolkit = Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').exists():
        env['CUDA_HOME'] = str(toolkit)
        env['PATH'] = str(toolkit/'bin')+os.pathsep+env['PATH']
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=1', '-m', 'scripts.toolmaze_agent.train_gigpo', '--config', args.config]
    if args.resume:
        command += ['--resume', args.resume]
    receipt = {'state': 'starting', 'gpu': args.gpu, 'launcher_pid': os.getpid(),
               'command': command, 'started_at_unix': time.time(), 'original_source_sha256': hashes}
    write_json(report/'launch.json', receipt)
    process = None
    try:
        with (report/'train.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(state='running', trainer_pid=process.pid)
            write_json(report/'launch.json', receipt)
            code = process.wait()
            receipt.update(state='complete' if code == 0 else 'failed', returncode=code)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        receipt.update(ended_at_unix=time.time(),
                       original_sources_unchanged=all(digest(ROOT/p) == h for p, h in hashes.items()))
        write_json(report/'launch.json', receipt)
    if receipt.get('returncode'):
        raise SystemExit(receipt['returncode'])


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128+signum))
    main()
