"""Mirror a detached OBC campaign and announce completion or monitoring failure."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from .experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args(); root = args.local.resolve()
    lock = (root/'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote = json.loads((root/'deployment.json').read_text())['remote']
    env = os.environ.copy()
    if not env.get('SSHPASS'):
        env['SSHPASS'] = subprocess.check_output(['security','find-generic-password','-s',
                                                 'autodl-password','-w'],text=True).strip()
    deadline = time.time()+args.hours*3600
    message = ''
    try:
        while True:
            subprocess.run(['rsync','-az','--exclude=*.tmp','--exclude=__pycache__',
                '-e','sshpass -e ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3',
                'autodl:'+remote+'/',str(root)+'/'],env=env,check=True,timeout=300)
            status = json.loads((root/'obc_status.json').read_text())
            data = dict(state='monitoring',pid=os.getpid(),updated_unix=time.time(),remote_status=status)
            if status['state'] in ('completed','failed'):
                data['state']='remote_'+status['state']
                if status['state']=='completed':
                    final = root/'main'/status['checkpoint']
                    if hashlib.sha256(final.read_bytes()).hexdigest()!=status['sha256']:
                        raise RuntimeError('Mirrored checkpoint hash mismatch')
                    if not json.loads((root/'final_verification.json').read_text())['passed']:
                        raise RuntimeError('Final verification missing or failed')
                message='OBC training '+status['state']+'. Results and logs mirrored locally.'
            elif time.time()>deadline:
                data['state']='monitoring_window_ended'
                message='OBC monitoring window ended; last remote state: '+status['state']
            elif time.time()-status['updated_unix']>300:
                # Diagnose a stale supervisor without changing the remote process.
                check=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl',
                    'kill -0 '+shlex.quote(str(int(status['pid'])))],env=env,capture_output=True,text=True,timeout=30)
                if check.returncode:
                    raise RuntimeError('Supervisor heartbeat stale; process check failed: '+check.stderr)
            write_json(root/'monitor_status.json',data)
            print(json.dumps(data),flush=True)
            if message:
                break
            time.sleep(60)
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        write_json(root/'monitor_status.json',dict(state='monitoring_failed',error=error,updated_unix=time.time()))
        message='OBC monitoring failed; remote computation status is unconfirmed. See monitor_status.json.'
        raise
    finally:
        (root/'MONITOR_NOTICE.md').write_text('# OBC monitoring notice\n\n'+message+'\n')
        subprocess.run(['osascript','-e','on run argv\ndisplay notification (item 1 of argv) with title "OBC training"\nend run',message],capture_output=True,timeout=20)
        print(message,flush=True)


if __name__ == '__main__':
    main()
