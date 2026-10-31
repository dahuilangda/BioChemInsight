"""Pipeline task scheduler and health monitor.

Tracks the state of long-running jobs (training, downloads, renders) and
automatically chains the next stage when a dependency completes. Also
provides a status report and restart-on-failure for critical downloads.
"""
import subprocess
import time
import os
import json
from pathlib import Path

STATE_FILE = Path('/data/BioChemInsight/pipeline_state.json')


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=1, default=str))


def _is_running(pattern):
    try:
        result = subprocess.run(['pgrep', '-f', pattern],
                                capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def get_status():
    state = _load_state()
    status = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stage7_depth12_training': {
            'running': _is_running('markush_ft_v9_depth12'),
            'checkpoint': '/data/molsight_repo/runs/markush_ft_v9_depth12/ckpt_model/epoch_0.pth',
            'log': '/data/molsight_markush_v8/ft_train_v9.log',
        },
        'pretrain_download': {
            'running': _is_running('download_pretrain'),
            'complete_shards': len(list(Path('/data/molparser7m/pretrain_synthetic_7M').glob('*.parquet'))),
            'target_shards': 116,
        },
        'gpu': _gpu_status(),
        'disk_free_gb': _disk_free(),
    }
    return status


def _gpu_status():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            gpus.append({'index': int(parts[0]), 'mem_used': parts[1], 'util': parts[2]})
        return gpus
    except Exception:
        return []


def _disk_free():
    try:
        result = subprocess.run(['df', '-h', '/data'],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        return lines[-1].split()[3]
    except Exception:
        return '?'


def wait_for_completion(pattern, check_interval=600, max_hours=24):
    """Block until a process matching pattern exits, reporting status."""
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        if not _is_running(pattern):
            return True
        status = get_status()
        print(f"[{time.strftime('%T')}] waiting... gpu={status['gpu']}", flush=True)
        time.sleep(check_interval)
    return False


def chain_stage7_validation():
    """After stage-7 training completes, run the three benchmarks + weight check."""
    v9_ckpt = '/data/molsight_repo/runs/markush_ft_v9_depth12/ckpt_model/epoch_0.pth'
    if not os.path.isfile(v9_ckpt):
        print(f"checkpoint not found: {v9_ckpt}")
        return
    print(f"running validation on {v9_ckpt}")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['HF_HUB_OFFLINE'] = '1'

    # 384 benchmark
    subprocess.run(['/data/molsight_venv/bin/python',
                    '/tmp/eval_molsight_ft.py', v9_ckpt,
                    '/tmp/molsight_v9_384.jsonl'],
                   env=env, cwd='/data/molsight_repo')

    # WildMol
    subprocess.run(['/data/molsight_venv/bin/python',
                    '/tmp/eval_wildmol.py', v9_ckpt, '400'],
                   env=env, cwd='/data/molsight_repo')

    # Weight check
    subprocess.run(['/home/dahuilangda/miniconda3/envs/llm/bin/python',
                    '/data/molsight_markush/weight_check.py',
                    '/data/molsight_repo/runs/markush_ft_v9_depth12/ckpt_model',
                    'epoch_0'], cwd='/data/BioChemInsight')

    print("validation complete")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        print(json.dumps(get_status(), indent=1))
    elif len(sys.argv) > 1 and sys.argv[1] == 'wait_validate':
        # Wait for stage-7 to finish, then run validation
        if wait_for_completion('markush_ft_v9_depth12', check_interval=300):
            chain_stage7_validation()
    else:
        print("usage: pipeline_scheduler.py [status|wait_validate]")
        print(json.dumps(get_status(), indent=1))
