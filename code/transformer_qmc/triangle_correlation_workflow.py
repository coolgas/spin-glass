"""Sequential finite validation, size-frontier and dense-field pilot phases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for phase in ('validation', 'frontier', 'scan'):
        record = dict(phase=phase, state='running')
        (args.output/'workflow.json').write_text(json.dumps(record, indent=2)+'\n')
        with (args.output/f'{phase}.log').open('ab') as stream:
            result = subprocess.run([sys.executable, '-u', '-m',
                'transformer_qmc.triangle_correlation_dmrg', '--output', str(args.output),
                '--phase', phase], stdout=stream, stderr=subprocess.STDOUT)
        record['state'] = 'complete' if result.returncode == 0 else 'blocked'
        (args.output/'workflow.json').write_text(json.dumps(record, indent=2)+'\n')
        from .triangle_correlation_report import report
        report(args.output)
        if result.returncode:
            print((args.output/'status.json').read_text(), flush=True)
            raise SystemExit(result.returncode)
        print(f'{phase} completed', flush=True)


if __name__ == '__main__':
    main()
