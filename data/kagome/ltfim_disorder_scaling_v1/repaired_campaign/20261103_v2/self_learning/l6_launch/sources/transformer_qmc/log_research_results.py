"""Persist Markdown research snapshots independently of the local mirror."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import time


STAGES=('notebook_stage_b','notebook_stage_b_escalation')


def atomic_text(path, text):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def snapshot(root):
    output=root/'research_logs';output.mkdir(exist_ok=True)
    states={};now=datetime.now(timezone.utc)
    for stage in STAGES:
        folder=root/stage
        if not (folder/'manifest.json').exists():continue
        manifest=json.loads((folder/'manifest.json').read_text())
        tasks=json.loads((folder/'tasks.json').read_text())
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'state':'initialized'}
        states[stage]=status['state']
        records=[];digests={}
        for task in tasks:
            path=folder/'results'/(task['id']+'.json')
            if not path.exists():continue
            raw=path.read_bytes();record=json.loads(raw)
            if record['task']!=task or record['signature']!=manifest['signature'] or record['state']!='completed':
                raise RuntimeError('Result identity mismatch: '+str(path))
            records.append(record);digests[task['id']]=hashlib.sha256(raw).hexdigest()
        failures=sorted(p.name for p in (folder/'failures').glob('*.json'))
        deadline=status.get('deadline_unix',manifest['deadline_unix'])
        evidence=dict(signature=manifest['signature'],state=status['state'],results_sha256=digests,failures=failures,
            process_deadline_unix=deadline)
        digest=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
        stamp=output/(stage+'.snapshot.json')
        if stamp.exists() and json.loads(stamp.read_text()).get('digest')==digest:continue
        lines=['# '+stage+' — research result log','',
            'Recorded UTC: '+now.isoformat(),'',
            '**System: L=4, N=48 spins, periodic boundaries in both directions.**',
            '',f"State: **{status['state']}**. Completed results: **{len(records)}/{len(tasks)}**. Recorded failures: **{len(failures)}**.",
            '', 'Manifest signature: `'+manifest['signature']+'`.',
            '', 'Active process deadline UTC: '+datetime.fromtimestamp(deadline,timezone.utc).isoformat()+'.',
            '', 'Original manifest deadline UTC: '+datetime.fromtimestamp(manifest['deadline_unix'],timezone.utc).isoformat()+'.',
            '', 'All results are feasibility/development evidence. Ground-state convergence and phase-boundary inference remain unqualified. An ancestry alarm means any cycle has descendant ESS below 8 or largest-lineage fraction above 0.5. Every alarm is retained.',
            '', '## Settings and completion','',
            '| Level | Gamma | h | Guide | W | Equilibration | Measurement | Forward | Cycles | Branch | Done / expected | Alarms | Median seconds | Minimum ESS |',
            '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        groups=defaultdict(list);expected=defaultdict(list)
        def key(t):return (t.get('ladder','base'),t['gamma'],t['h'],t['method'])
        for task in tasks:expected[key(task)].append(task)
        for record in records:groups[key(record['task'])].append(record)
        for k,group in sorted(expected.items()):
            p=group[0]['parameters'];done=groups[k]
            median=f"{statistics.median(r['metrics']['seconds'] for r in done):.4g}" if done else 'pending'
            ess=f"{min(r['minimum_descendant_ess'] for r in done):.4g}" if done else 'pending'
            lines.append('| '+' | '.join(map(str,(*k,p['n_walkers'],p['n_equilibrate']*p['tau_block'],
                p['n_measure']*p['tau_block'],p['tau_forward'],p['n_cycles'],p['tau_block'],
                f'{len(done)}/{len(group)}',sum(r['ancestry_alarm'] for r in done),median,ess)))+' |')
        lines+=['','## Individual population results','',
            'Values below are individual population estimates, without uncertainty bars. Forward cycles are correlated diagnostics, not independent repeats. Compare four independent repeats within each disorder before forming disorder averages. The escalation subset contains one disorder and cannot estimate disorder uncertainty.',
            '', '| Task ID | Disorder | Repeat | Level | Gamma | h | Guide | E/N | q2 | chi_conn/N | Seconds | ESS minimum | Largest lineage | Alarm |',
            '|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
        for r in records:
            t=r['task'];v=r['result']
            lines.append('| '+' | '.join(map(str,(t['id'],t['sample_id'],t['repeat'],t.get('ladder','base'),
                t['gamma'],t['h'],t['method'],*(f'{v[o]:.8g}' for o in ('energy_per_site','q2','chi_connected_per_site')),
                f"{r['metrics']['seconds']:.5g}",f"{r['minimum_descendant_ess']:.5g}",f"{r['max_lineage']:.5g}",r['ancestry_alarm'])))+' |')
        lines+=['','## Continuation and interpretation','',
            'The authorized next step is to finish the base pilot, then execute the fixed 144-run population/time escalation schedule. Its controller waits for successful base completion and acquires the shared supervisor lock. W256/W1024, separate equilibration4 and forward2 tests, and h=0.05 diagnose population loss, projection dependence and cost.',
            '', 'Ancestry recovery alone does not qualify physical observables. Stage C still requires measured feasible cost, eight development disorders, one-parameter convergence comparisons, alternative starts and combined checks. Any unresolved context remains unqualified. Final anchor scope and acceptance standards are identical for L=4,6,8,10.',
            '', 'At the recorded deadline workers checkpoint and pause; the instance stays on. Machine-readable source results and full cycle diagnostics are in `../'+stage+'/results/`. Snapshot provenance stores SHA-256 for every included result.','']
        text='\n'.join(lines)
        archive=output/'history';archive.mkdir(exist_ok=True)
        name=now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+stage+'_'+digest[:12]
        atomic_text(archive/(name+'.md'),text)
        atomic_text(archive/(name+'.json'),json.dumps(evidence,indent=2))
        atomic_text(output/(stage+'.md'),text)
        atomic_text(stamp,json.dumps(dict(digest=digest,recorded_utc=now.isoformat(),completed=len(records),state=status['state']),indent=2))
    atomic_text(output/'logger_status.json',json.dumps(dict(updated_utc=now.isoformat(),states=states),indent=2))
    return states


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=180)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--launch',type=Path,help='Separate continuation launch record')
    args=parser.parse_args()
    if args.interval<=0:raise ValueError('interval must be positive')
    output=args.root/'research_logs';output.mkdir(exist_ok=True)
    with (output/'logger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=json.loads((args.launch or args.root/'operational_launch.json').read_text())['deadline_unix']
        while True:
            states=snapshot(args.root)
            print(datetime.now(timezone.utc).isoformat(),states,flush=True)
            last=states.get(STAGES[-1],'')
            terminal=last.startswith('awaiting_') or last.startswith('paused') or last=='failed'
            if args.once or terminal or time.time()>deadline+900:return
            time.sleep(args.interval)


if __name__=='__main__':main()
