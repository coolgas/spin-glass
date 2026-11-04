"""Bounded notebook Section 5 escalations after completion of its base pilot.

Representative selection is fixed by field and disorder ID, not observable
agreement. This controller never promotes feasibility outputs to final data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def build_tasks():
    from transformer_qmc.production_runner import make_task,settings
    # Low gamma stresses ancestry; high gamma stresses cost. Compare zero field
    # to the notebook's required small nonzero field. Keep one fixed disorder
    # for the first bounded escalation; Stage C still requires eight disorders.
    levels=[('baseline_h005',64,2.,1.),('population256',256,2.,1.),
            ('population1024',1024,2.,1.),('equilibration4',1024,4.,1.),
            ('forward2',1024,2.,2.)]
    tasks=[]
    for label,w,eq,fw in levels:
        for repeat in range(4):
            for gamma in (.5,2.):
                for h in ((.05,) if label=='baseline_h005' else (0.,.05)):
                    for method in ('network','constant'):
                        tasks.append(make_task('notebook_stage_b_escalation',4,60000,gamma,h,
                            method,'float64',repeat,settings(w,eq,1.,fw,4,.1),ladder=label))
    assert len(tasks)==144 and len({t['id'] for t in tasks})==144
    return tasks


def initialize(root):
    from transformer_qmc.production_protocol import identity,source_hashes
    from transformer_qmc.experiment import write_json
    parent=json.loads((root/'notebook_stage_b/manifest.json').read_text())
    if identity({k:v for k,v in parent.items() if k!='signature'})!=parent['signature']:
        raise RuntimeError('Parent manifest identity mismatch')
    source=source_hashes(root/'runner_source/code')
    if source!=parent['source_sha256']:raise RuntimeError('Frozen scientific source changed')
    launch=json.loads((root/'operational_launch.json').read_text())
    if hashlib.sha256(Path(launch['checkpoint']).read_bytes()).hexdigest()!=parent['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    output=root/'notebook_stage_b_escalation';output.mkdir(exist_ok=True)
    tasks=build_tasks()
    manifest=dict(schema=1,stage='notebook_stage_b_escalation',notebook_section=5,
        parent_signature=parent['signature'],source_sha256=source,
        checkpoint_sha256=parent['checkpoint_sha256'],notebook_sha256=parent['notebook_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        support_sha256=hashlib.sha256((root/'notebook_stage_b_support.py').read_bytes()).hexdigest(),
        tasks_sha256=identity(tasks),task_count=len(tasks),deadline_unix=launch['deadline_unix'],
        selection=dict(L=4,disorders=[60000],gamma=[.5,2.],h=[0.,.05],repeats=4,
            rationale='Prespecified low/high gamma cost and ancestry extremes; zero versus small nonzero h; first fixed development disorder. No selection by observable agreement.'),
        comparisons=[['base_or_baseline_h005','population256'],['population256','population1024'],
                     ['population1024','equilibration4'],['population1024','forward2']],
        baseline_reuse='At h=0 reuse notebook_stage_b L4 sample60000 gamma0.5/2 W64, all four repeats and both methods.',
        gate_policy='Complete base pilot first. Escalations measure cost and ancestry only. No automatic Stage C or final-data admission.',
        stage_c_pending=dict(populations=[512,2048,8192],equilibration=[16,32,64],
            forward=[4,8,16],branch=[.1,.05],cycles=[16,32],development_disorders=8,
            required='Priced feasible budget, difficult-disorder selection by runtime/ancestry, one-parameter ladders, combined stress test, alternative starts, Section 7 uncertainty criteria and 49-context qualification.'),
        final_data_started=False,ground_state_qualified=False)
    manifest['signature']=identity(manifest)
    for name,obj in [('manifest.json',manifest),('tasks.json',tasks)]:
        p=output/name
        if p.exists():
            if json.loads(p.read_text())!=obj:raise RuntimeError('Existing identity mismatch: '+name)
        else:write_json(p,obj)
    for folder in ['results','checkpoints','heartbeats','failures']:(output/folder).mkdir(exist_ok=True)
    return output,manifest,tasks,launch


def verify_parent(root):
    from transformer_qmc.production_runner import verified_result
    from transformer_qmc.production_protocol import identity
    p=root/'notebook_stage_b';manifest=json.loads((p/'manifest.json').read_text())
    tasks=json.loads((p/'tasks.json').read_text())
    assert identity({k:v for k,v in manifest.items() if k!='signature'})==manifest['signature']
    assert identity(tasks)==manifest['tasks_sha256']
    from transformer_qmc.production_runner import make_task,settings
    expected=[make_task('notebook_stage_b',4,sid,g,h,m,'float64',r,settings(64,2.,1.,1.,4,.1))
              for r in range(4) for sid in range(60000,60004) for g in (.5,1.,2.)
              for h in (0.,.15,.3) for m in ('network','constant')]
    assert tasks==expected, 'Parent does not match notebook Section 5'
    if list((p/'failures').glob('*.json')) or (p/'PAUSE.json').exists():
        raise RuntimeError('Base pilot has unresolved failures/pause')
    complete=all(verified_result(p,t,manifest['signature']) for t in tasks)
    status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
    if status.get('state')=='failed':raise RuntimeError('Base supervisor failed')
    return complete and status.get('state') in ('awaiting_stage_b_cost_review','awaiting_notebook_stage_b_review')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--initialize-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    sys.path.insert(0,str(root/'runner_source/code'))
    import notebook_stage_b_support as support
    from transformer_qmc.experiment import write_json
    signal.signal(signal.SIGTERM,support.terminate);signal.signal(signal.SIGINT,support.terminate)
    import fcntl
    controller_lock=(root/'research_followup.lock').open('a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output,manifest,tasks,launch=initialize(root)
    if args.initialize_only:
        print('INITIALIZED_ESCALATIONS',len(tasks),manifest['signature'],flush=True);return
    try:
        while not verify_parent(root):
            if support.STOP or time.time()>=launch['deadline_unix']:
                write_json(output/'status.json',dict(state='paused_before_prerequisites',updated_unix=time.time(),pid=os.getpid(),final_data_started=False));return
            write_json(output/'status.json',dict(state='waiting_for_base_pilot',pid=os.getpid(),updated_unix=time.time(),expected=len(tasks),completed=0,deadline_unix=launch['deadline_unix'],final_data_started=False))
            time.sleep(30)
        # Wait for the prior supervisor's lock release without overlapping GPU work.
        with (root/'pipeline/supervisor.lock').open('a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if support.STOP or time.time()>=launch['deadline_unix']:
                        write_json(output/'status.json',dict(state='paused_waiting_for_lock',pid=os.getpid(),updated_unix=time.time(),final_data_started=False));return
                    time.sleep(5)
            assert verify_parent(root)
            if support.STOP or time.time()>=launch['deadline_unix']:
                write_json(output/'status.json',dict(state='paused_before_dispatch',pid=os.getpid(),updated_unix=time.time(),final_data_started=False));return
            write_json(output/'status.json',dict(state='starting_notebook_stage_b_escalation',pid=os.getpid(),updated_unix=time.time(),expected=len(tasks),deadline_unix=launch['deadline_unix'],final_data_started=False))
            write_json(root/'active_stage.json',dict(stage='notebook_stage_b_escalation',
                status_path='notebook_stage_b_escalation/status.json',pid=os.getpid(),updated_unix=time.time()))
            support.supervise(root,output,manifest,tasks,launch,stage_name='notebook_stage_b_escalation')
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        write_json(output/'status.json',dict(state='failed',pid=os.getpid(),error=error,updated_unix=time.time(),final_data_started=False))
        raise


if __name__=='__main__':main()
