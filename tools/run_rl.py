#!/usr/bin/env python3
"""Own one local vLLM sampler and a DeepSpeed/TRL LoRA training process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import load_config, read_json, write_json


def require_idle(indices):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                   '--format=csv,noheader'], text=True).splitlines()
    available = {}
    for line in rows.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(',')]
        available[int(index)] = uuid not in jobs and int(memory) < 128
    if any(not available.get(index, False) for index in indices):
        raise RuntimeError('A selected GPU is occupied; refusing to overlap another task')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True, help='Training GPU indices in PCI bus order')
    parser.add_argument('--sampler-gpu', required=True)
    parser.add_argument('--config', default='scripts/toolmaze_agent/configs/rl_lora.yaml')
    parser.add_argument('--resume', help='Native checkpoint including DeepSpeed optimizer and RNG state')
    args = parser.parse_args()
    train_gpus = [int(x) for x in args.gpus.split(',')]
    sampler_gpu = int(args.sampler_gpu)
    if len(set(train_gpus+[sampler_gpu])) != len(train_gpus)+1:
        parser.error('Sampler and training GPUs must be distinct')
    require_idle(train_gpus+[sampler_gpu])
    config = load_config(args.config)
    name = Path(config['output']).name
    reports = ROOT / 'reports/rl' / name
    reports.mkdir(parents=True, exist_ok=True)
    address = urlparse(config['training']['vllm_server_base_url'])
    if address.hostname != '127.0.0.1':
        raise ValueError('Managed sampler must bind to loopback')
    with socket.socket() as sock:
        if sock.connect_ex((address.hostname, address.port)) == 0:
            raise RuntimeError('Sampler port already occupied; refusing to reuse an unknown service')
    env = dict(os.environ)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               VLLM_SERVER_DEV_MODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_CACHE_ROOT=str(ROOT/'.cache/vllm'),
               TRITON_CACHE_DIR=str(ROOT/'.cache/triton_sft'),
               TORCH_EXTENSIONS_DIR=str(ROOT/'.cache/torch_extensions'))
    env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    # Global HTTP proxies must not intercept the trainer's loopback sampler RPCs.
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, [env.get(key,''), '127.0.0.1', 'localhost']))
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'.deps/rl'), str(ROOT/'.deps/sft_fast'),
                                      str(ROOT), env.get('PYTHONPATH','')])
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    toolkit = Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').exists():
        env['CUDA_HOME'] = str(toolkit)
        env['PATH'] = str(toolkit/'bin')+os.pathsep+env['PATH']
    server_env = dict(env, CUDA_VISIBLE_DEVICES=str(sampler_gpu))
    server_command = [sys.executable, '-u', '-m', 'vllm.entrypoints.cli.main', 'serve',
        str((ROOT/config['model']).resolve()), '--host', address.hostname, '--port', str(address.port),
        '--dtype', 'bfloat16', '--generation-config', 'vllm', '--max-model-len', '8192',
        '--gpu-memory-utilization', str(config.get('sampler_memory_utilization', 0.8)),
        '--max-num-seqs', '128', '--max-num-batched-tokens', '16384',
        '--limit-mm-per-prompt', json.dumps({'image':0,'video':0}),
        '--compilation-config', json.dumps({'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes':[1,2,4,8,16,32,64,96,128]}),
        '--weight-transfer-config', json.dumps({'backend':'nccl'}), '--logprobs-mode', 'processed_logprobs',
        '--max-logprobs', '-1']
    train_command = [sys.executable,'-u','-m','torch.distributed.run','--standalone',
        f'--nproc_per_node={len(train_gpus)}','-m','scripts.toolmaze_agent.train_rl','--config',args.config]
    if args.resume:
        train_command += ['--resume',args.resume]
    receipt = {'state':'starting_sampler','gpus':train_gpus,'sampler_gpu':sampler_gpu,
               'config':config,'server_command':server_command,'training_command':train_command,
               'launcher_pid':os.getpid(),'started_at_unix':time.time()}
    write_json(reports/'launch.json',receipt)
    server = trainer = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with (reports/'sampler.log').open('a') as slog, (reports/'train.log').open('a') as tlog:
            server = subprocess.Popen(server_command,cwd=ROOT,env=server_env,
                stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
            receipt['server_pid']=server.pid
            write_json(reports/'launch.json',receipt)
            started=time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Sampler exited with {server.returncode}; inspect sampler.log')
                try:
                    with opener.open(config['training']['vllm_server_base_url']+'/health',timeout=2) as response:
                        if response.status==200:
                            break
                except OSError:
                    pass
                if time.monotonic()-started>360:
                    raise TimeoutError('Sampler initialization exceeded 360 seconds')
                time.sleep(2)
            require_idle(train_gpus)
            trainer = subprocess.Popen(train_command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=args.gpus),
                stdout=tlog,stderr=subprocess.STDOUT,start_new_session=True)
            receipt.update(state='training',trainer_pid=trainer.pid)
            write_json(reports/'launch.json',receipt)
            print(f'LoRA GRPO started; train={args.gpus}, sampler={sampler_gpu}, logs={reports}',flush=True)
            code=trainer.wait()
            receipt.update(state='complete' if code==0 else 'failed',returncode=code,ended_at_unix=time.time())
            write_json(reports/'launch.json',receipt)
            if code:
                raise RuntimeError(f'Training exited with {code}; inspect train.log')
    except BaseException as error:
        receipt.update(state='failed',error=f'{type(error).__name__}: {error}',ended_at_unix=time.time())
        write_json(reports/'launch.json',receipt)
        progress = ROOT/config['output']/'progress.json'
        if progress.exists():
            write_json(progress,{**read_json(progress),'state':'failed','error':receipt['error'],
                                'ended_at_unix':receipt['ended_at_unix']})
        raise
    finally:
        for process in (trainer,server):
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)


if __name__=='__main__':
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
