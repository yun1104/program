#!/usr/bin/env python3
"""Merge training, rollout, checkpoint and process state into a live read-only view."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    try:
        data = Path(f'/proc/{int(pid)}/stat').read_text()
        return data.rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, ValueError, TypeError):
        return False


def snapshot(run, report):
    progress = read(run/'progress.json')
    rollout = read(run/'rollout_progress.json')
    launch = read(report/'launch.json')
    step = progress.get('step', 0)
    total = progress.get('total_steps', launch.get('config', {}).get('training', {}).get('max_steps', 100))
    metrics = []
    try:
        for line in (report/'train.log').read_text().splitlines():
            if line.startswith("{'loss':"):
                try:
                    metrics.append({k:float(v) for k,v in ast.literal_eval(line).items()})
                except (ValueError, SyntaxError, TypeError):
                    pass
    except OSError:
        pass
    checkpoints = []
    for path in run.glob('checkpoint-*/checkpoint_complete.json'):
        marker = read(path)
        if marker.get('optimizer_and_rng_saved'):
            checkpoints.append(marker['step'])
    running = alive(launch.get('trainer_pid'))
    state = 'complete' if step >= total or launch.get('state') == 'complete' else 'running' if running else 'failed' if launch.get('state') == 'failed' or progress.get('state') == 'failed' else 'stopped'
    phase = state
    if running:
        if rollout.get('step') == step and rollout.get('completed', 0) < rollout.get('total', 0):
            phase = 'sampling'
        elif rollout.get('step') == step:
            phase = 'scoring_and_backpropagation'
        else:
            phase = 'checkpoint_or_weight_sync'
    # Historical log entries survive native resume; use recent complete online
    # steps, excluding replay and the current incomplete update, for the estimate.
    timings = [m['step_time'] for m in metrics
               if m.get('epoch', 0) > 0 and m.get('step_time', 0) > 0][-5:]
    remaining = (total-step)*sum(timings)/len(timings) if timings and state == 'running' else None
    now = datetime.now(TZ)
    value = {'updated_at':now.isoformat(timespec='seconds'), 'state':state,
        'phase':phase, 'completed_steps':step, 'total_steps':total,
        'current_update':step+1 if state == 'running' else None,
        'percent_complete':round(100*step/total,1),
        'rollout':rollout, 'latest_logged_metrics':metrics[-1] if metrics else {},
        'latest_resumable_checkpoint':max(checkpoints) if checkpoints else None,
        'trainer_alive':running,'sampler_alive':alive(launch.get('server_pid')),
        'estimated_remaining_hours':round(remaining/3600,2) if remaining else None,
        'estimated_finish_beijing':(now+timedelta(seconds=remaining)).isoformat(timespec='minutes') if remaining else None,
        'estimate_note':'Recent five complete online steps; trajectory length changes the estimate.',
        'refresh_seconds':5}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/step_credit_full_20261027')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    run = ROOT/args.run
    report = ROOT/'reports/step_credit'/run.name
    while not (run/'manifest.json').exists():
        if read(report/'launch.json').get('state') == 'failed':
            return
        time.sleep(2)
    while True:
        value = snapshot(run, report)
        target = run/'live_progress.json'
        temporary = target.with_suffix('.json.monitor.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(target)
        if args.once or value['state'] in ('complete','failed','stopped'):
            print(json.dumps(value,ensure_ascii=False),flush=True)
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
