"""Detached OBC supervisor: CUDA/resume checks, pilot, main, and durable report."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .experiment import load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    lock = (root/'obc_supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'code'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')

    def report(state, **extra):
        status = dict(state=state, pid=os.getpid(), started_unix=started, updated_unix=time.time(), **extra)
        write_json(root/'obc_status.json', status)
        lines = ['# OBC shared-guide training campaign', '', f'State: **{state}**', '',
                 'Separate OBC-v1 architecture and checkpoint namespace; all runs use open kagome patches.',
                 'Recipe: fresh 600-step batch-128 pilot, then 6000-step batch-256 main initialized from pilot.',
                 'Sizes 2,4,6,8 contribute equally to every VMC step; fields (0.5,0), (0.5,0.3), (2,0.3).',
                 'Exact references are evaluation-only. Training completion does not qualify GFMC convergence.', '']
        for phase in ('smoke_full','smoke_resume','capacity_128','capacity_256','pilot','main'):
            p = root/phase/'status.json'
            if p.exists():
                lines += [f'## {phase}', '', '```json', p.read_text().strip(), '```', '']
        if extra:
            lines += ['## Supervisor details', '', '```json', json.dumps(extra,indent=2), '```', '']
        temp = root/'OBC_RESEARCH.tmp'
        temp.write_text('\n'.join(lines)); temp.replace(root/'OBC_RESEARCH.md')

    def train(phase, folder, steps, batch, *extra):
        command = [sys.executable, '-B', '-u', '-m', 'transformer_qmc.train_obc',
                   '--output', str(root/folder), '--phase', phase, '--steps', str(steps),
                   '--batch', str(batch), *extra]
        with (root/(folder+'.log')).open('a') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    report(folder+'_running', child_pid=child.pid)
            if code:
                raise subprocess.CalledProcessError(code, command)

    try:
        report('cuda_and_restart_validation')
        with (root/'tests.log').open('w') as log:
            subprocess.run([sys.executable,'-B','-m','unittest','transformer_qmc.test_obc'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Same total schedule and initial RNG: stop at step 1, then restore and finish.
        train('smoke','smoke_full',3,8,'--skip-evaluation')
        train('smoke','smoke_resume',3,8,'--skip-evaluation','--stop-after','1')
        train('smoke','smoke_resume',3,8,'--skip-evaluation')
        prefix = 'obc_v1_shared_alibi__smoke__L02-04-06-08__seed501__final.pt'
        a = load_checkpoint(root/'smoke_full'/prefix,'cpu')
        b = load_checkpoint(root/'smoke_resume'/prefix,'cpu')
        max_difference = 0.
        for name, value in a['model'].items():
            if not torch.allclose(value,b['model'][name],rtol=1e-6,atol=1e-7):
                raise RuntimeError('CUDA resume mismatch: '+name)
            max_difference = max(max_difference,float((value-b['model'][name]).abs().max()))
        evidence = dict(passed=True, maximum_parameter_difference=max_difference,
                        compared_steps=3, interrupted_after=1, sizes=[2,4,6,8],fields=3)
        write_json(root/'restart_validation.json',evidence)
        report('batch_capacity_validation',restart_validation=evidence)
        train('smoke','capacity_128',3,128,'--skip-evaluation')
        train('smoke','capacity_256',3,256,'--skip-evaluation')
        report('pilot_training',restart_validation=evidence)
        train('pilot','pilot',600,128)
        pilot = root/'pilot/obc_v1_shared_alibi__pilot__L02-04-06-08__seed501__final.pt'
        pilot_status = json.loads((root/'pilot/status.json').read_text())
        if pilot_status['state'] != 'completed':
            raise RuntimeError('Pilot did not complete validation')
        report('main_training',pilot_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest())
        train('main','main',6000,256,'--initialize',str(pilot))
        final_status = json.loads((root/'main/status.json').read_text())
        if final_status['state'] != 'completed':
            raise RuntimeError('Main did not complete validation')
        report('final_checkpoint_verification')
        with (root/'verification.log').open('a') as log:
            subprocess.run([sys.executable,'-B','-m','transformer_qmc.verify_obc_checkpoint',
                            '--root',str(root)], env=env, stdout=log,
                           stderr=subprocess.STDOUT,check=True)
        report('completed',checkpoint=final_status['checkpoint'],sha256=final_status['sha256'],
               elapsed_seconds=time.time()-started)
    except BaseException:
        report('failed',error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
