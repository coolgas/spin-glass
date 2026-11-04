"""Resume the frozen Stage B escalation within a separate operational window.

Stage this file at the campaign root alongside notebook_stage_b_support.py.
Scientific manifests, task seeds, and the original launch record are preserved.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def validate(root):
    from transformer_qmc.production_protocol import identity, source_hashes
    from transformer_qmc.production_runner import verified_result
    import research_followup
    output = root/'notebook_stage_b_escalation'
    manifest = json.loads((output/'manifest.json').read_text())
    tasks = json.loads((output/'tasks.json').read_text())
    original = json.loads((root/'operational_launch.json').read_text())
    if identity({k: v for k, v in manifest.items() if k != 'signature'}) != manifest['signature']:
        raise RuntimeError('Manifest signature mismatch')
    if identity(tasks) != manifest['tasks_sha256'] or tasks != research_followup.build_tasks():
        raise RuntimeError('Frozen task identity mismatch')
    if source_hashes(root/'runner_source/code') != manifest['source_sha256']:
        raise RuntimeError('Frozen scientific source changed')
    for path, expected in [(Path(original['checkpoint']), manifest['checkpoint_sha256']),
                           (root/'notebook_stage_b_support.py', manifest['support_sha256']),
                           (root/'research_followup.py', manifest['script_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen dependency changed: '+str(path))
    if not research_followup.verify_parent(root):
        raise RuntimeError('Base pilot prerequisites incomplete')
    if (output/'PAUSE.json').exists() or list((output/'failures').glob('*.json')):
        raise RuntimeError('Unreviewed pause or failure')
    pending = [t for t in tasks if not verified_result(output, t, manifest['signature'])]
    for task in pending:
        path = output/'checkpoints'/(task['id']+'.json.gz')
        if path.exists():
            with gzip.open(path, 'rt') as handle:
                saved = json.load(handle)
            if saved['task'] != task or saved['signature'] != manifest['signature']:
                raise RuntimeError('Pending checkpoint identity mismatch')
    return output, manifest, tasks, original, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 24:
        raise ValueError('Continuation window must be within 24 hours')
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'runner_source/code'))
    import notebook_stage_b_support as support
    from transformer_qmc.experiment import write_json
    with (root/'research_followup.lock').open('a') as controller, \
         (root/'pipeline/supervisor.lock').open('a') as supervisor:
        for lock in (controller, supervisor):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output, manifest, tasks, original, pending = validate(root)
        print(json.dumps(dict(validated=True, completed=len(tasks)-len(pending),
                              pending=len(pending), signature=manifest['signature'])), flush=True)
        if args.validate_only:
            return
        started = time.time()
        window = root/'continuations'/time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started))
        window.mkdir(parents=True, exist_ok=False)
        launch = dict(checkpoint=original['checkpoint'], started_unix=started,
            deadline_unix=started+args.hours*3600, original_launch=original,
            manifest_signature=manifest['signature'], pending_task_ids=[t['id'] for t in pending],
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            authorization='Finish paused ancestry/time checks, then assess physical-observable convergence.',
            estimated_remaining_gpu_hours=17.3, window_hours=args.hours)
        write_json(window/'launch.json', launch)
        write_json(window/'previous_status.json', json.loads((output/'status.json').read_text()))
        write_json(root/'active_continuation.json', dict(launch_path=str(window/'launch.json')))
        write_json(root/'active_stage.json', dict(stage='notebook_stage_b_escalation',
            status_path='notebook_stage_b_escalation/status.json', pid=os.getpid(), updated_unix=started))
        (output/'supervisor.pid').write_text(str(os.getpid())+'\n')
        write_json(output/'status.json', dict(state='starting_notebook_stage_b_escalation',
            pid=os.getpid(), updated_unix=started, deadline_unix=launch['deadline_unix'],
            completed=len(tasks)-len(pending), expected=len(tasks), final_data_started=False))
        signal.signal(signal.SIGTERM, support.terminate)
        signal.signal(signal.SIGINT, support.terminate)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root/'research_validation/code')+os.pathsep+str(root/'runner_source/code')
        logger_file = (window/'logger.log').open('a')
        logger = subprocess.Popen([sys.executable, '-u', str(root/'log_research_results.py'),
            '--root', str(root), '--launch', str(window/'launch.json')],
            stdout=logger_file, stderr=subprocess.STDOUT, start_new_session=True)
        (root/'research_logs/logger.pid').write_text(str(logger.pid)+'\n')
        failure = None
        try:
            support.supervise(root, output, manifest, tasks, launch,
                              stage_name='notebook_stage_b_escalation')
        except BaseException as exc:
            failure = repr(exc)
            write_json(output/'status.json', dict(state='failed', pid=os.getpid(),
                updated_unix=time.time(), error=failure, final_data_started=False))
            raise
        finally:
            with (window/'assessment.log').open('a') as log:
                assessment = subprocess.run([sys.executable, '-B', '-m',
                    'transformer_qmc.assess_physical_convergence', '--root', str(root)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if logger.poll() is None:
                logger.terminate()
                logger.wait(timeout=30)
            subprocess.run([sys.executable, str(root/'log_research_results.py'),
                '--root', str(root), '--once'], check=True, timeout=120)
            write_json(window/'outcome.json', dict(finished_unix=time.time(),
                status=json.loads((output/'status.json').read_text()), error=failure,
                assessment_returncode=assessment.returncode))
            logger_file.close()
            print('CONTINUATION_TERMINAL', (window/'outcome.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
