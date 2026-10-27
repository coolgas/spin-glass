"""Detached sequential GPU campaigns; rerunning resumes both from saved state."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from .experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaigns = [('fixed_size','transformer_qmc.experiment',800),
                 ('unified_alibi','transformer_qmc.unified_experiment',4800)]
    status = {}
    for name,module,steps in campaigns:
        destination = args.output/name
        if (destination/'training_finished.json').exists():
            status[name] = dict(state='completed')
            continue
        command = [sys.executable,'-B','-u','-m',module,'--output',str(destination),
                   '--steps',str(steps),'--batch','128']
        status[name] = dict(state='running',started=datetime.now(timezone.utc).isoformat(),command=command)
        write_json(args.output/'status.json',status)
        with (args.output/f'{name}.log').open('a',buffering=1) as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        status[name].update(state='completed' if result.returncode==0 else 'failed',
                            returncode=result.returncode,finished=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'status.json',status)
        if result.returncode:
            raise RuntimeError(f'{name} failed with code {result.returncode}; see {name}.log')
    write_json(args.output/'status.json',status)


if __name__ == '__main__':
    main()
