"""Wait for the finite auxiliary pilot, then export an audited final report."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .collective_campaign import write


def main(root, output, pid):
    record_path = output/'completion_report_status.json'
    while True:
        status = json.loads((root/'status.json').read_text())
        if status['state'] == 'completed_unqualified':
            from .notebook_auxiliary_results import build
            build(root, output)
            write(record_path, dict(state='audited_report_completed',
                utc=datetime.now(timezone.utc).isoformat(), campaign_status=status))
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            from .notebook_auxiliary_results import build
            build(root, output)
            write(record_path, dict(state='worker_exited_before_completion',
                utc=datetime.now(timezone.utc).isoformat(), campaign_status=status))
            raise RuntimeError('Pilot worker exited before the finite queue completed; partial report saved')
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main(args.root, args.output, args.pid)
