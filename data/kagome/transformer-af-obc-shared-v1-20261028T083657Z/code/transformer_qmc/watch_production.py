"""Read-only AutoDL mirroring and execution audit; never modifies remote jobs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def atomic(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def terminal_state(root, active, status):
    state=status.get('state','')
    terminal=state.startswith('paused') or state in ('failed','qualification_complete_pending_review') or state.startswith('awaiting_')
    pending=root/'notebook_stage_b_escalation/status.json'
    if active['stage']=='notebook_stage_b' and state.startswith('awaiting_') and pending.exists():
        followup=json.loads(pending.read_text())
        if followup.get('state') in ('waiting_for_base_pilot','starting_notebook_stage_b_escalation'):
            # A dead controller must not keep this monitor waiting indefinitely.
            return time.time()-followup.get('updated_unix',0)>180
    pointer=root/'active_continuation.json'
    if terminal and active['stage']=='notebook_stage_b_escalation' and pointer.exists():
        launch=Path(json.loads(pointer.read_text())['launch_path'])
        window=root/'continuations'/launch.parent.name
        outcome=window/'outcome.json'
        if outcome.exists():
            final=json.loads(outcome.read_text())
            if (final.get('finished_unix',0)>=status.get('updated_unix',0) and
                final.get('status',{}).get('state')==state and
                final.get('status',{}).get('deadline_unix')==status.get('deadline_unix')):
                return True
        if time.time()-status.get('updated_unix',0)>900:
            raise RuntimeError('Remote compute is terminal, but continuation outcome/report finalization is missing after 15 minutes')
        return False
    return terminal


def notify(root, data, enabled):
    """Persist terminal notice; optionally request a macOS desktop notification."""
    state=data['state']
    if state=='failed':
        message='GFMC monitoring failed. Remote compute status is unconfirmed; see mirror_status.json.'
    elif state=='finished_monitoring_window':
        message='GFMC monitoring window ended. See mirror_status.json for the last remote state.'
    else:
        message=f"GFMC {data.get('supervisor_state','terminal')}: {data.get('supervisor_completed','?')}/144 escalation tasks. Reports saved locally."
        pointer=root/'active_continuation.json'
        if pointer.exists():
            window=Path(json.loads(pointer.read_text())['launch_path']).parent.name
            outcome=root/'continuations'/window/'outcome.json'
            if outcome.exists():
                final=json.loads(outcome.read_text())
                data['assessment_returncode']=final.get('assessment_returncode')
                if final.get('assessment_returncode')!=0:
                    message+=' Physical assessment failed; inspect continuation assessment.log.'
    notice=dict(message=message,created_unix=time.time(),monitor_status=data)
    (root/'MONITOR_NOTICE.md').write_text('# GFMC monitoring notice\n\n'+message+'\n')
    if enabled and sys.platform=='darwin':
        try:
            result=subprocess.run(['osascript','-e',
                'on run argv\ndisplay notification (item 1 of argv) with title "GFMC research"\nend run',
                message],capture_output=True,text=True,timeout=20)
            notice['desktop_notification_returncode']=result.returncode
            notice['desktop_notification_stderr']=result.stderr
        except Exception as exc:
            notice['desktop_notification_error']=repr(exc)
    atomic(root/'monitor_notice.json',notice)
    print(message,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',default='/root/autodl-tmp/transformer-af-production-20261027-v1')
    parser.add_argument('--local',type=Path,default=Path('data/kagome/transformer_af_production_v1'))
    parser.add_argument('--interval',type=int,default=180)
    parser.add_argument('--hours',type=float,default=12.25)
    parser.add_argument('--transfer-timeout',type=int,default=900)
    parser.add_argument('--audit-timeout',type=int,default=300)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--notify',action='store_true',help='Request a desktop notification on terminal or monitoring failure')
    args=parser.parse_args();args.local=args.local.resolve();args.local.mkdir(parents=True,exist_ok=True)
    if min(args.interval,args.hours,args.transfer_timeout,args.audit_timeout)<=0:
        raise ValueError('positive monitoring interval, window and timeouts required')
    lock=(args.local/'mirror.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (args.local/'mirror.pid').write_text(str(os.getpid())+'\n')
    env=os.environ.copy()
    if not env.get('SSHPASS'):
        env['SSHPASS']=subprocess.check_output(['security','find-generic-password','-s','autodl-password','-w'],text=True).strip()
    if not env.get('SSHPASS'):raise RuntimeError('keychain credential unavailable')
    started=time.time();deadline=started+args.hours*3600
    rsync=['rsync','-az','--exclude=__pycache__','--exclude=*.tmp','--exclude=*.tmp.gz',
             '--exclude=*.tmp.npz','-e','sshpass -e ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3']
    while True:
        try:
            data=dict(state='syncing_results',pid=os.getpid(),updated_unix=time.time(),
                      interval_seconds=args.interval,deadline_unix=deadline)
            atomic(args.local/'mirror_status.json',data)
            subprocess.run(rsync+['--exclude=/mirror*','--exclude=/monitor_launch.json',
                '--exclude=/monitor_notice.json','--exclude=/MONITOR_NOTICE.md',
                '--exclude=/pipeline/checkpoints/','--exclude=/notebook_stage_b/checkpoints/',
                '--exclude=/notebook_stage_b_escalation/checkpoints/',
                'autodl:'+args.remote+'/',str(args.local)+'/'],env=env,check=True,timeout=args.transfer_timeout)
            data.update(state='auditing_results',updated_unix=time.time())
            atomic(args.local/'mirror_status.json',data)
            report=subprocess.run([sys.executable,'-B','-m','transformer_qmc.report_production','--root',str(args.local)],
                                  env=env,text=True,capture_output=True,timeout=args.audit_timeout)
            print(report.stdout,flush=True)
            if report.returncode:raise RuntimeError(report.stdout+report.stderr)
            if (args.local/'PHYSICAL_CONVERGENCE.md').exists():
                assessment=subprocess.run([sys.executable,'-B','-m','transformer_qmc.assess_physical_convergence',
                    '--root',str(args.local)],env=env,text=True,capture_output=True,timeout=args.audit_timeout)
                print(assessment.stdout,flush=True)
                if assessment.returncode:raise RuntimeError(assessment.stdout+assessment.stderr)
            pointer=args.local/'active_stage.json'
            active=json.loads(pointer.read_text()) if pointer.exists() else {'status_path':'pipeline/status.json','stage':'preliminary'}
            status=json.loads((args.local/active['status_path']).read_text())
            data.update(state='syncing_checkpoints',updated_unix=time.time(),
                        active_stage=active['stage'],supervisor_state=status.get('state'),supervisor_completed=status.get('completed'))
            atomic(args.local/'mirror_status.json',data)
            checkpoints=args.local/'pipeline/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
            subprocess.run(rsync+['autodl:'+args.remote+'/pipeline/checkpoints/',str(checkpoints)+'/'],
                           env=env,check=True,timeout=args.transfer_timeout)
            for stage in ('notebook_stage_b','notebook_stage_b_escalation'):
                if not (args.local/stage/'manifest.json').exists():continue
                checkpoints=args.local/stage/'checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
                subprocess.run(rsync+['autodl:'+args.remote+'/'+stage+'/checkpoints/',str(checkpoints)+'/'],
                               env=env,check=True,timeout=args.transfer_timeout)
            data.update(state='mirroring',updated_unix=time.time(),checkpoint_backup_unix=time.time())
            terminal=terminal_state(args.local,active,status)
            if terminal:data['state']='finished_remote_'+status['state']
            elif args.once:data['state']='one_shot_complete'
            elif time.time()>=deadline:data['state']='finished_monitoring_window'
            atomic(args.local/'mirror_status.json',data)
            if terminal or time.time()>=deadline:notify(args.local,data,args.notify)
            if terminal or args.once or time.time()>=deadline:return
        except BaseException:
            error=traceback.format_exc();print(error,flush=True)
            failure=dict(state='failed',error=error,pid=os.getpid(),updated_unix=time.time(),
                         scope='local monitoring; remote compute state unconfirmed')
            atomic(args.local/'mirror_status.json',failure)
            notify(args.local,failure,args.notify)
            raise
        time.sleep(args.interval)


if __name__=='__main__':
    main()
