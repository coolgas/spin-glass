"""One-shot throughput estimate; execution time is not convergence time."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text())


def describe(record):
    counts=record['plan']; estimates=record['estimates']
    budget=("**Runtime policy:** no wall-clock cutoff. The finite campaign is authorized to continue until all planned tasks have been attempted. "
        "The throughput estimates above retain their original measurement timestamp." if record['deadline_utc'] is None else
        f"**Current dispatch cutoff:** {record['deadline_utc']}, approximately {record['hours_until_deadline']:.2f} hours after this snapshot. "
        "The full plan requires an extended/resumed budget; this estimate does not extend the deadline. Already active GFMC trajectories may drain afterward.")
    lines=['## Expanded campaign: scope and runtime estimate', '',
        f"**Measured at {record['snapshot_utc']}.** Runtime scenarios, not a completion guarantee or a convergence certificate.", '',
        f"Both L=4 (N=48) and L=6 (N=108) target Gamma=0.55,...,0.70 by 0.01, plus 0.85 and 1.00, at h=0.15. Eight fixed disorders, both guide families, two training seeds, and two projection repeats are retained.", '',
        f"**Full plan:** {counts['fits']} fits; {counts['main_runs']} main GFMC runs + {counts['audit_runs']} population/time/start checks = {counts['projector_runs']} runs.", '',
        f"**Saved progress:** {record['completed_fits']} completed fits, {record['completed_runs']} completed GFMC records; {len(record['failures'])} recorded failures.", '',
        '| Milestone | Planning time from this snapshot |', '|---|---|',
        '| First L=6, Gamma=0.55 RBM main repeats | Approximately 20–40 minutes; individual-disorder runs |',
        '| Gamma=0.55, both sizes/guides, all eight disorders: main ensemble | Approximately 4–6 hours |',
        '| Gamma=0.55 including all planned sensitivities | Approximately 6–10 hours |',
        '| Gamma=0.55–0.60, both sizes, including sensitivities | Approximately 1–2 days |',
        '| Entire expanded plan including 0.85 and 1.00 | Approximately 4–6 days of continued operation |', '',
        '**Measured rates and assumptions:**']
    for row in record['training_rates']:
        lines.append(f"- L={row['L']} {row['kind']}: {row['central_seconds']/60:.1f} minutes per fit; {row['basis']}.")
    for row in record['projection_rates']:
        if row['L']==6 and row['kind']=='rbm_2':
            lines.append(f"- L=6 RBM GFMC {row['audit']}: {row['seconds']/60:.1f} minutes per run; {row['basis']} ({row['observations']} observations).")
    lines.extend(['',f"At {record['training_workers']} training workers, central full-plan training workload is {estimates['training_hours']:.1f} wall-hours; the explicit training-rate scenario spans {estimates['training_low_hours']:.1f}–{estimates['training_high_hours']:.1f} hours.",
        f"At {record['projector_workers']} GFMC workers, central full-plan projection workload is {estimates['projector_hours']:.1f} wall-hours. Training and projection overlap, so these totals are not added.",
        f"With completed work credited, the idealized remaining-work bottleneck is {estimates['remaining_ideal_bottleneck_hours']:.1f} hours; the planning scenario is {estimates['planning_low_hours']:.1f}–{estimates['planning_high_hours']:.1f} hours, allowing 15–60% pipeline/throughput overhead and the upper training-rate scenario.", '',
        '**Limitations:** This is a saved forecast for the initial expanded-grid restart, not an adaptive scheduler or live ETA. The milestone ranges are planning judgments based on these throughput estimates. L6 RBM projection timing is extrapolated from saved partial repaired trajectories. L4 estimates use repaired completions/checkpoints when available. L6 Jastrow projection timings use retired-campaign timing metadata only, not its physics results. Field, disorder, guide and CPU contention can change runtime. Before-projection guide sampling is covered by the overhead scenario. Training samples are still few; imported pilot timing is not treated as representative of four-worker throughput. Failed sampling checks can leave contexts unqualified rather than supply a complete ensemble. The 4–6 day range covers the prescribed runs only; further convergence extensions have no defensible fixed ETA yet.', '',
        budget])
    return '\n'.join(lines)+'\n'


def estimate(root):
    output=root/'results'; manifest=read(output/'manifest.json'); launch=read(root/'launch.json')
    status=read(output/'parallel_status.json'); now=datetime.now(timezone.utc)
    fits=[]; completed=[]; checkpoints=[]; legacy=[]
    for path in sorted((output/'training').glob('*.json')):
        row=read(path)
        if row.get('status')=='completed' and 'signature' in row:
            fits.append(dict(task=row['task'],seconds=row['seconds'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in sorted((output/'projector/runs').glob('*.json')):
        row=read(path);completed.append(dict(task=row['task'],seconds=row['seconds'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    completed_tasks={json.dumps(row['task'],sort_keys=True) for row in completed}
    for path in sorted((output/'projector/checkpoints').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as data:meta=json.loads(str(data['metadata']))
        if json.dumps(meta['task'],sort_keys=True) in completed_tasks:continue
        task=meta['task'];total=round((task['equil']+task['cycles']*task['forward'])/task['delta'])
        if meta['step']<=0:continue
        checkpoints.append(dict(task=task,step=meta['step'],total=total,seconds=meta['seconds'],
            projected_seconds=meta['seconds']*total/meta['step'],source=str(path)))
    legacy_root=root.parent/'disorder-scaling-20261103-v1/results/projector/runs'
    for path in sorted(legacy_root.glob('*.json')):
        row=read(path);task=row['task']
        if task['L']==6 and task['kind']=='jastrow':
            legacy.append(dict(task=task,seconds=row['seconds'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rates=[]
    # L4 repaired fits measure the current four-worker training regime.
    for kind in manifest['kinds']:
        values=[r['seconds'] for r in fits if r['task']['L']==4 and r['task']['kind']==kind]
        central=statistics.median(values)
        rates.append(dict(L=4,kind=kind,central_seconds=central,low_seconds=.9*central,
            high_seconds=1.2*central,basis=f'{len(values)} repaired completed fits; scenario -10%/+20%'))
    # No completed current-allocation L6 fits yet: label the planning envelope.
    for kind in manifest['kinds']:
        rates.append(dict(L=6,kind=kind,central_seconds=22.5*60,low_seconds=15*60,high_seconds=30*60,
            basis='explicit 15–30 minute planning assumption; partial L6 Jastrow optimizer timing is about 15 minutes before frozen diagnostics'))
    projection_rates=[]; main_count=0;audit_count=0;projection_seconds=0.;fit_count=0
    for L in manifest['sizes']:
        fields=manifest['fields_by_size'][str(L)];fit_count+=len(fields)*len(manifest['disorders'])*len(manifest['kinds'])*len(manifest['seeds'])
        for kind in manifest['kinds']:
            for audit in ['main','population','time','start']:
                exact=[r['seconds'] for r in completed if (r['task']['L'],r['task']['kind'],r['task']['audit'])==(L,kind,audit)]
                partial=[r['projected_seconds'] for r in checkpoints if (r['task']['L'],r['task']['kind'],r['task']['audit'])==(L,kind,audit)]
                if exact:seconds=statistics.median(exact);basis='repaired completed trajectories';n=len(exact)
                elif partial:seconds=statistics.median(partial);basis='repaired partial-trajectory extrapolation';n=len(partial)
                elif L==6 and kind=='jastrow':
                    historical=[r['seconds'] for r in legacy if r['task']['audit']==audit]
                    seconds=statistics.median(historical);basis='legacy timing only; no physical result reused';n=len(historical)
                else:
                    main=[r['seconds'] for r in completed if (r['task']['L'],r['task']['kind'],r['task']['audit'])==(L,kind,'main')]
                    if not main:main=[r['projected_seconds'] for r in checkpoints if (r['task']['L'],r['task']['kind'],r['task']['audit'])==(L,kind,'main')]
                    seconds=statistics.median(main)*(2 if audit in ['population','time'] else 1);basis='repaired main timing scaled by planned walker/time factor';n=len(main)
                count=len(fields)*(len(manifest['disorders']) if audit=='main' else len(manifest['audit_ids']))*2
                if audit=='main':main_count+=count
                else:audit_count+=count
                projection_seconds+=seconds*count
                projection_rates.append(dict(L=L,kind=kind,audit=audit,seconds=seconds,basis=basis,observations=n,planned_runs=count))
    training_seconds={key:sum(row[key]*len(manifest['fields_by_size'][str(row['L'])])*len(manifest['disorders'])*len(manifest['seeds']) for row in rates) for key in ['central_seconds','low_seconds','high_seconds']}
    trainers=launch['training_workers'];workers=launch['projector_workers']
    training_hours=training_seconds['central_seconds']/trainers/3600;projection_hours=projection_seconds/workers/3600
    earned_fit_seconds=sum(next(row['central_seconds'] for row in rates if (row['L'],row['kind'])==(fit['task']['L'],fit['task']['kind'])) for fit in fits)
    earned_projection_seconds=sum(r['seconds'] for r in completed)+sum(r['seconds'] for r in checkpoints)
    remaining_training=max(0,training_seconds['central_seconds']-earned_fit_seconds)/trainers/3600
    remaining_projection=max(0,projection_seconds-earned_projection_seconds)/workers/3600
    bottleneck=max(remaining_training,remaining_projection)
    hours_remaining=None if status['deadline_unix'] is None else max(0,(datetime.fromtimestamp(status['deadline_unix'],timezone.utc)-now).total_seconds()/3600)
    return dict(snapshot_utc=now.isoformat(),deadline_utc=launch['deadline_utc'],hours_until_deadline=hours_remaining,
        fields_by_size=manifest['fields_by_size'],training_workers=trainers,projector_workers=workers,
        completed_fits=len(fits),completed_runs=len(completed),failures=status['failures'],
        plan=dict(fits=fit_count,main_runs=main_count,audit_runs=audit_count,projector_runs=main_count+audit_count),
        training_rates=rates,projection_rates=projection_rates,
        estimates=dict(training_hours=training_hours,training_low_hours=training_seconds['low_seconds']/trainers/3600,training_high_hours=training_seconds['high_seconds']/trainers/3600,
            projector_hours=projection_hours,remaining_ideal_bottleneck_hours=bottleneck,planning_low_hours=1.15*bottleneck,
            planning_high_hours=max(1.6*bottleneck,training_seconds['high_seconds']/trainers/3600)),
        evidence=dict(corrected_fits=fits,corrected_completed_projections=completed,corrected_partial_projections=checkpoints,legacy_timing_only=legacy))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();record=estimate(args.campaign);args.destination.mkdir(parents=True,exist_ok=True)
    (args.destination/'timing_estimate.json').write_text(json.dumps(record,indent=2)+'\n')
    text=describe(record);(args.destination/'TIMING.md').write_text(text);print(text)


if __name__=='__main__':main()
