"""Detached development-only selection and bounded shared-guide continuations.

Run from a new frozen remote source tree. Existing campaigns are read-only.
No reserved test disorders are inspected or used for selection.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .compare_low_field import compare
from .experiment import write_json


def selection_score(run):
    """Predeclared score; lower is better, with a high-field retention gate."""
    low = [v for v in run['rows'] if v['gamma'] <= 1.]
    high = [v for v in run['rows'] if (v['gamma'], v['h']) == (2., .3)]
    infidelity = float(np.mean([max(0., 1-v['fidelity']) for v in low]))
    correlation = float(np.mean([v['connected_correlation_rms_error'] for v in low]))
    chi_error = float(np.mean([abs(v['chi_connected']-v['exact_chi']) for v in low]))
    high_min = min(v['fidelity'] for v in high)
    score = infidelity + .25*correlation + .02*chi_error
    return dict(score=score, mean_low_field_infidelity=infidelity,
                mean_low_field_C_rms=correlation, mean_low_field_absolute_chi_error=chi_error,
                lowest_high_field_fidelity=high_min, high_field_retained=high_min >= .98)


def await_campaign(directory):
    launches = json.loads((directory/'launches.json').read_text())
    while True:
        rows = []
        for launch in launches['runs']:
            path = directory/launch['name']/'status.json'
            status = json.loads(path.read_text()) if path.exists() else dict(state='starting')
            proc = Path('/proc')/str(launch['pid'])/'stat'
            alive = proc.exists() and proc.read_text().split()[2] != 'Z'
            if not alive and status['state'] != 'completed':
                raise RuntimeError(f'Predecessor failed: {launch["name"]}: {status}')
            rows.append(dict(name=launch['name'], status=status, alive=alive))
        if all(row['status']['state'] == 'completed' for row in rows):
            return rows
        print('WAIT',json.dumps(rows),flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--main-results', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'overnight.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output/'launches.json').exists():
        raise RuntimeError('Continuations already launched; inspect existing PIDs before restarting')
    source_root = Path(__file__).resolve().parents[2]
    policy = dict(development_ids=list(range(10000,10008)), reserved_test_ids=list(range(40000,40008)),
                  selection='mean low-field infidelity + 0.25 mean off-diagonal connected-C RMS '
                  '+ 0.02 mean absolute connected susceptibility error; '
                  'lowest high-field fidelity >=0.98 preferred',
                  low_fields='all measured gamma <=1 including h=0.15 scan',
                  fallback='If every candidate misses the high-field gate, retain that fact and '
                  'choose the smallest score; no accuracy qualification is inferred.',
                  steps=args.steps, seeds=[701,702,703,704],
                  exact_labels_used_for_training=False,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')})
    write_json(args.output/'selection_policy.json', policy)
    write_json(args.output/'supervisor_status.json', dict(state='waiting_for_predecessor'))
    rows = await_campaign(args.predecessor)
    candidates = []
    recipes = {}
    for row in rows:
        directory = args.predecessor/row['name']
        checkpoint = directory/row['status']['checkpoint']
        candidates.append(checkpoint)
        recipes[str(checkpoint)] = json.loads((directory/'campaign.json').read_text())
    # Main final is compared on validation IDs only; final-test evaluation is not read.
    while True:
        status=json.loads((args.main_results/'status.json').read_text())
        if status['state']=='completed': break
        print('WAIT_MAIN',status,flush=True)
        time.sleep(60)
    main_checkpoint=args.main_results/status['checkpoint']
    candidates += [main_checkpoint,args.initial]
    main_recipe=dict(method='adam',continuous=False,architecture=dict(completion=False))
    recipes[str(main_checkpoint)]=main_recipe
    recipes[str(args.initial)]=main_recipe
    write_json(args.output/'supervisor_status.json', dict(state='development_comparison'))
    comparison = compare(candidates, ids=tuple(policy['development_ids']), field_scan=True)
    write_json(args.output/'development_comparison.json', comparison)
    ranked = []
    for run in comparison['runs']:
        ranked.append(dict(checkpoint=run['checkpoint'],sha256=run['sha256'],
                           recipe=recipes[run['checkpoint']],**selection_score(run)))
    ranked.sort(key=lambda v:(not v['high_field_retained'],v['score']))
    chosen = ranked[0]
    def family(candidate):
        recipe=candidate['recipe']
        return (recipe['method'],recipe['continuous'],bool(recipe['architecture'].get('completion',False)))
    alternative=next((v for v in ranked if family(v)!=family(chosen)),ranked[1])
    decision=dict(policy=policy, ranked=ranked, selected=chosen,
                  alternative=alternative, reserved_tests_inspected=False,
                  interpretation='Development selection only; eight disorders do not certify disorder generalization.')
    write_json(args.output/'selection.json', decision)
    plans=[('selected701',701,chosen,None),('selected702',702,chosen,None),
           ('conservative_sr703',703,chosen,.025),('alternative704',704,alternative,None)]
    launches=[]
    environment=dict(os.environ,PYTHONPATH=str(source_root/'code'),OMP_NUM_THREADS='3',
                     OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
    for name,seed,candidate,rate in plans:
        recipe=candidate['recipe']
        method='sr' if rate else recipe['method']
        command=[sys.executable,'-B','-u','-m','transformer_qmc.train_remedies',
                 '--output',str(args.output/name),'--initialize',candidate['checkpoint'],
                 '--method',method,'--seed',str(seed),'--steps',str(args.steps),
                 '--batch','1024','--high-batch','512','--score-chunk','64',
                 '--validate-every','300','--context-offset','600']
        if recipe['continuous']: command.append('--continuous')
        if recipe['architecture'].get('completion',False): command.append('--completion')
        if rate: command += ['--rate',str(rate)]
        with (args.output/(name+'.log')).open('a',buffering=1) as log:
            process=subprocess.Popen(command,cwd=source_root,env=environment,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        launches.append(dict(name=name,pid=process.pid,command=command,
                             initialization_sha256=candidate['sha256']))
        write_json(args.output/'partial_launches.json',dict(runs=launches))
    write_json(args.output/'launches.json',dict(runs=launches))
    write_json(args.output/'supervisor_status.json', dict(state='continuations_running',runs=launches))
    await_campaign(args.output)
    write_json(args.output/'supervisor_status.json', dict(state='continuations_completed'))
    print('Overnight continuations completed; reserved test IDs remain untouched.',flush=True)


if __name__ == '__main__':
    main()
