"""Mirror the matched GFMC benchmark and refresh its executed report."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if not os.environ.get('SSHPASS'):
        raise RuntimeError('SSHPASS must be loaded from environment or Keychain')
    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results/'mirror.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        try:
            while True:
                subprocess.run([
                    'rsync', '-az', '--exclude', '*.tmp', '-e',
                    'sshpass -e ssh -o ConnectTimeout=15',
                    'autodl:/root/autodl-tmp/transformer-af-ltfim-20261027-pure-vs-guided-v1/results/',
                    str(args.results)+'/'
                ], check=True, timeout=120)
                status = json.loads((args.results/'status.json').read_text())
                subprocess.run([
                    sys.executable, '-m', 'transformer_qmc.report_pure_gfmc',
                    '--results', str(args.results), '--notebook', str(args.notebook)
                ], check=True, timeout=300)
                finished = status['state'] in ('completed', 'failed', 'time_cap_reached')
                state = dict(state='completed' if finished else 'mirroring',
                             remote_status=status, updated_unix=time.time(),
                             notebook=str(args.notebook.resolve()))
                write_json(args.results/'mirror_status.json', state)
                print(json.dumps(state), flush=True)
                if finished:
                    return
                if time.time()-started > 3*3600:
                    write_json(args.results/'mirror_status.json', dict(state, state='mirror_time_cap'))
                    return
                time.sleep(args.interval)
        except BaseException as error:
            write_json(args.results/'mirror_status.json', dict(
                state='failed', error=repr(error), updated_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
