"""Queue the bounded direct GFMC pilot after the scan releases the GPU."""

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
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def status(state, **extra):
        write_json(args.output/'supervisor.json', dict(state=state, pid=os.getpid(),
                   updated_unix=time.time(), started_unix=started,
                   pilot_time_cap_hours=4, **extra))

    status('queued_for_scan_gpu')
    while True:
        summary = args.scan/'sampled_summary.json'
        if summary.exists() and json.loads(summary.read_text())['completed']:
            break
        supervisor = args.scan/'supervisor.json'
        if supervisor.exists() and json.loads(supervisor.read_text())['state'] == 'failed':
            status('blocked_by_scan_failure')
            return
        if time.time()-started > 8*3600:
            status('queue_time_cap')
            return
        time.sleep(30)
    with (args.output/'pilot.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'transformer_qmc.direct_gfmc_pilot',
                                    '--checkpoint', str(args.checkpoint), '--output', str(args.output)],
                                   stdout=log, stderr=subprocess.STDOUT)
        status('running', child_pid=process.pid)
        try:
            code = process.wait(timeout=4*3600)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            status('pilot_time_cap_reached', resumable=True)
            return
        status('completed' if code == 0 else 'failed', exit_code=code)


if __name__ == '__main__':
    main()
