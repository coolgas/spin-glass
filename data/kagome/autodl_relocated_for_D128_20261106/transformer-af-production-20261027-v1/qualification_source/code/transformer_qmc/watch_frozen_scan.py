"""Local detached result mirror and executed-report refresh for the remote campaign.

Credentials stay in SSHPASS, loaded by the caller from Keychain when necessary.
Failures stop this watcher with the full command error preserved in its log.
"""

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
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if not os.environ.get('SSHPASS'):
        raise RuntimeError('SSHPASS must be loaded from environment or Keychain')
    args.results.mkdir(parents=True, exist_ok=True)
    lock = (args.results/'mirror.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote = '/root/autodl-tmp/'
    stages = [('transformer-af-ltfim-20261027-frozen-scan-v1', args.results),
              ('transformer-af-ltfim-20261027-direct-preview-v1', args.results/'direct_preview'),
              ('transformer-af-ltfim-20261027-direct-gfmc-v1', args.results/'direct_pilot')]
    started = time.time()
    while True:
        for campaign, destination in stages:
            destination.mkdir(exist_ok=True)
            subprocess.run(['rsync', '-az', '--exclude', '*.tmp', '--exclude', '*.tmp.npz',
                            '-e', 'sshpass -e ssh -o ConnectTimeout=15',
                            f'autodl:{remote}{campaign}/results/', str(destination)+'/' ],
                           check=True, timeout=180)
        subprocess.run([sys.executable, '-m', 'transformer_qmc.report_frozen_scan',
                        '--results', str(args.results), '--notebook', str(args.notebook),
                        '--direct-preview', str(args.results/'direct_preview'),
                        '--direct-pilot', str(args.results/'direct_pilot')], check=True, timeout=600)
        scan = json.loads((args.results/'supervisor.json').read_text())
        pilot_path = args.results/'direct_pilot/supervisor.json'
        pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
        finished = (scan['state'] in ('completed', 'failed') and
                    pilot.get('state') in ('completed', 'failed', 'pilot_time_cap_reached',
                                           'queue_time_cap', 'blocked_by_scan_failure'))
        state = dict(state='completed' if finished else 'mirroring', updated_unix=time.time(),
                     scan_state=scan['state'], direct_pilot_state=pilot.get('state'),
                     notebook=str(args.notebook.resolve()))
        write_json(args.results/'mirror_status.json', state)
        print(json.dumps(state), flush=True)
        if finished:
            return
        if time.time()-started > 13*3600:
            write_json(args.results/'mirror_status.json', dict(state, **{'state': 'mirror_time_cap'}))
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
