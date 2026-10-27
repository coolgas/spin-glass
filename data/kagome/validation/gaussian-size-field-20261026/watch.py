"""Watch this campaign, fetch immutable results, and verify them locally."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/autodl-tmp/gaussian-size-field-20261026'
PYTHON = '/root/miniconda3/envs/yjye_project_env/bin/python'


def ssh(command):
    return subprocess.run(['sshpass', '-e', 'ssh', '-o', 'ConnectTimeout=15', 'autodl', command],
                          check=True, text=True, capture_output=True, timeout=45).stdout


def main():
    if not os.environ.get('SSHPASS'):
        os.environ['SSHPASS'] = subprocess.run(
            ['security', 'find-generic-password', '-s', 'autodl-password', '-w'],
            check=True, text=True, capture_output=True).stdout.strip()
    start = time.monotonic()
    while True:
        code = ("import json; from pathlib import Path; p=Path("+repr(REMOTE)+
                "); print(json.dumps({n:json.loads((p/n).read_text()) for n in "
                "('launch.json','progress.json','exit.json') if (p/n).exists()}))")
        state = json.loads(ssh(PYTHON+" -c "+shell_quote(code)))
        for name, value in state.items():
            (ROOT/name).write_text(json.dumps(value, indent=2)+'\n')
        count = state.get('progress.json', {}).get('completed', 0)
        print('PROGRESS', count, '/ 192', flush=True)
        if 'exit.json' in state:
            break
        if time.monotonic()-start > 12*3600:
            raise TimeoutError('12-hour watcher limit reached; detached remote job may still be running')
        time.sleep(60)
    code = ("import hashlib,json; from pathlib import Path; p=Path("+repr(REMOTE)+
            "); print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() "
            "for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts "
            "and f.name not in ('remote_sha256.json',) and not f.name.endswith('.tmp')}))")
    remote_hashes = json.loads(ssh(PYTHON+" -c "+shell_quote(code)))
    subprocess.run(['rsync', '-az', '--exclude=__pycache__', '-e',
                    'sshpass -e ssh -o ConnectTimeout=15', f'autodl:{REMOTE}/', str(ROOT)+'/'],
                   check=True, timeout=180)
    for name, digest in remote_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    (ROOT/'remote_sha256.json').write_text(json.dumps(remote_hashes, indent=2)+'\n')
    if state['exit.json']['exit_code']:
        print((ROOT/'failure.txt').read_text(), flush=True)
        raise RuntimeError('remote campaign failed; downloaded artifacts retained')
    subprocess.run([sys.executable, str(ROOT/'verify.py')], check=True)
    print('COMPLETE: downloaded, SHA256 checked, independently verified', flush=True)


def shell_quote(text):
    return "'"+text.replace("'", "'\"'\"'")+"'"


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT/'watch_failure.txt').write_text(traceback.format_exc())
        raise
