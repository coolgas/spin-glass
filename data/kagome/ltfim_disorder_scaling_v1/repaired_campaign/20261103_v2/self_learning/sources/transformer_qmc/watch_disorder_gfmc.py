"""Mirror the isolated disorder campaign and refresh its executed notebook."""

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
        raise RuntimeError('load SSHPASS from Keychain before launching')
    args.results.mkdir(parents=True, exist_ok=True)
    lock = (args.results/'mirror.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    remote = 'autodl:/root/autodl-tmp/transformer-af-ltfim-20261027-disorder-gfmc-v1/results/'
    failures = 0
    while True:
        try:
            subprocess.run(['rsync','-az','--exclude','*.tmp','--exclude','*.tmp.npz',
                            '-e','sshpass -e ssh -o ConnectTimeout=15',remote,
                            str(args.results)+'/' ],check=True,timeout=180)
            subprocess.run([sys.executable,'-m','transformer_qmc.report_disorder_gfmc',
                            '--results',str(args.results),'--notebook',str(args.notebook)],
                           check=True,timeout=600)
            production = json.loads((args.results/'status.json').read_text())
            rp = args.results/'reference_status.json'
            reference = json.loads(rp.read_text()) if rp.exists() else {}
            finished = production['state']=='completed' and reference.get('state')=='completed'
            state = dict(state='completed' if finished else 'mirroring',
                         production=production,reference=reference,updated_unix=time.time(),
                         notebook=str(args.notebook.resolve()))
            write_json(args.results/'mirror_status.json',state)
            print(json.dumps(state),flush=True)
            failures = 0
            if finished:
                from .audit_disorder_gfmc import audit
                result=audit(args.results)
                if not result['complete']:
                    raise RuntimeError(f'Incomplete final audit: {result}')
                return
            if production['state'] in ('failed','time_cap_reached'):
                raise RuntimeError(f"production stopped: {production}")
        except (subprocess.SubprocessError,OSError) as error:
            failures += 1
            print(repr(error),flush=True)
            if failures >= 5:
                raise
        if time.time()-started > 7*3600:
            raise RuntimeError('mirror exceeded seven hours')
        time.sleep(min(args.interval,60))


if __name__ == '__main__':
    main()
