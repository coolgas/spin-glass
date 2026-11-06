"""Mirror detached AutoDL correlation results and validate notebook attachments."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nbformat


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--interval',type=int,default=300)
    args=parser.parse_args()
    if not os.environ.get('SSHPASS'):
        raise RuntimeError('Load SSHPASS from the keychain before starting this mirror')
    args.local.mkdir(parents=True,exist_ok=True)
    while True:
        result=subprocess.run(['rsync','-az','--include=*/','--include=*.json',
            '--include=*.txt','--include=*.ipynb','--include=*.png','--include=*.pdf',
            '--exclude=*','-e','sshpass -e ssh -o ConnectTimeout=15',
            args.remote.rstrip('/')+'/',str(args.local)+'/'],capture_output=True,text=True)
        if result.returncode:
            print(result.stderr,flush=True)
        else:
            candidate=args.local/'results/report/triangle_correlation_scaling.ipynb'
            if candidate.exists():
                try:
                    notebook=nbformat.read(candidate,as_version=4)
                    nbformat.validate(notebook)
                    for cell in notebook.cells:
                        for name, attachment in cell.get('attachments',{}).items():
                            assert base64.b64decode(attachment['image/png']) == (candidate.parent/name).read_bytes()
                except (AssertionError, ValueError, OSError) as error:
                    print(f'Incomplete report transfer, retrying: {error}',flush=True)
                else:
                    shutil.copy2(candidate,Path('notebook/triangle_correlation_scaling.ipynb'))
            status=args.local/'results/resume.json'
            if status.exists():
                record=json.loads(status.read_text())
                if record.get('state')=='blocked' or (record.get('phase')=='scan' and record.get('state')=='complete'):
                    print(json.dumps(record),flush=True)
                    break
        time.sleep(args.interval)


if __name__=='__main__':
    main()
