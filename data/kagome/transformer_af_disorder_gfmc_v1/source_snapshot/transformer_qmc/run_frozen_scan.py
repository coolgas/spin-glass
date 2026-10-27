"""Detached campaign supervisor with explicit stage logs and a compute time cap."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started+8*3600
    base = [sys.executable, '-u', '-m', 'transformer_qmc.frozen_scan',
            '--checkpoint', str(args.checkpoint), '--output', str(args.output)]
    completed = []
    active = {}
    logs = []

    def save(state, **extra):
        write_json(args.output/'supervisor.json', dict(
            state=state, pid=os.getpid(), started_unix=started,
            completed_stages=completed, active={k: p.pid for k, p in active.items()},
            time_cap_hours=8, updated_unix=time.time(), **extra))

    def launch(stage, device):
        log = (args.output/f'{stage}.log').open('a')
        logs.append(log)
        process = subprocess.Popen(base+['--stage', stage, '--device', device],
                                   stdout=log, stderr=subprocess.STDOUT)
        active[stage] = process
        save('running')
        return process

    def finish(stage):
        code = active[stage].wait(timeout=max(1, deadline-time.time()))
        del active[stage]
        if code:
            raise RuntimeError(f'{stage} exited {code}; full error in {stage}.log')
        completed.append(stage)
        save('running')

    try:
        for stage in ('benchmark', 'exact', 'projection'):
            launch(stage, 'cuda')
            finish(stage)
        launch('gfmc', 'cpu')
        launch('sampled', 'cuda')
        finish('sampled')
        finish('gfmc')
        save('completed')
    except BaseException as error:
        for process in active.values():
            process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        save('failed', error=repr(error))
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
