"""Snapshot measured campaign timings; produce an explicitly conditional forecast."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import json
import shutil
import statistics

root = Path('/root/autodl-tmp/disorder-scaling-repaired-20261103-v2')
bundle = Path('/root/autodl-tmp/disorder-scaling-20261103-v1/preliminary_notebook_bundle')
dest = bundle/'data/kagome/ltfim_disorder_scaling_v1/repaired_campaign/20261103_v2'
archive = dest/'migration/timing_recalibration_20261103'
archive.mkdir(parents=True, exist_ok=True)
if not (archive/'TIMING_before.md').exists():
    shutil.copy2(dest/'TIMING.md', archive/'TIMING_before.md')
fits = [json.loads(p.read_text()) for p in (root/'results/training').glob('L*.json')]
runs = [json.loads(p.read_text()) for p in (root/'results/projector/runs').glob('*.json')]
status = json.loads((root/'results/parallel_status.json').read_text())
assert status['trainers'] == status['projector_workers'] == 2
assert status['deadline_unix'] is None
fit_groups = defaultdict(list)
run_groups = defaultdict(list)
for row in fits:
    t = row['task']
    # Imported pilot timings include different concurrency/partial continuation.
    if t['L'] == 6 and t['kind'] == 'rbm_2' and t['sample_id'] == 60000:
        continue
    fit_groups[(t['L'], t['kind'])].append(row['seconds'])
for row in runs:
    t = row['task']
    run_groups[(t['L'], t['kind'], t['audit'])].append(row['seconds'])
rates = []
for phase, groups in [('training', fit_groups), ('projection', run_groups)]:
    for key, seconds in sorted(groups.items()):
        population = fits if phase == 'training' else runs
        completed = sum(tuple(row['task'][k] for k in (['L', 'kind'] if phase == 'training' else ['L', 'kind', 'audit'])) == key for row in population)
        planned = 288 if phase == 'training' or key[2] == 'main' else 72
        median = statistics.median(seconds)
        rates.append(dict(phase=phase, group=list(key), n=len(seconds), completed=completed,
                          planned=planned, median_seconds=median, min_seconds=min(seconds),
                          max_seconds=max(seconds), remaining_worker_hours=(planned-completed)*median/3600))
work = {phase: sum(r['remaining_worker_hours'] for r in rates if r['phase'] == phase)
        for phase in ['training', 'projection']}
baseline = max(work.values())/2
restored = max(work['training']/4, work['projection']/6)
now = datetime.now(timezone.utc).isoformat()
record = dict(snapshot_utc=now, status=status, completed_fits=len(fits), completed_runs=len(runs),
              rates=rates, remaining_worker_hours=work,
              current_two_two_ideal_hours=baseline,
              current_planning_days=[baseline*1.15/24, baseline*1.60/24],
              conditional_four_six_ideal_hours=restored,
              overhead_factors=[1.15, 1.60],
              interpretation='Scenario bounds, not confidence intervals. Rates are mostly Gamma .55 under mixed historical concurrency. No automatic resource reassignment. Partial active work is not credited. No convergence ETA.',
              fits=[dict(task=r['task'],seconds=r['seconds']) for r in fits],
              runs=[dict(task=r['task'],seconds=r['seconds']) for r in runs])
(archive/'timing_snapshot.json').write_text(json.dumps(record, indent=2)+'\n')
lines = ['## Recalibrated campaign runtime', '',
         f'**Snapshot: {now}.** Supersedes the earlier 4–6 day forecast. All times below are remaining time from this snapshot.', '',
         f'Production has {len(fits)}/1152 completed fits and {len(runs)}/2016 completed GFMC records. Its live allocation is **two GPU training workers and two CPU GFMC workers**, with no deadline. Diagnostics have separate workers. Finishing diagnostics does not automatically enlarge the production pools.', '',
         '| Milestone | Planning estimate |', '|---|---|',
         '| Both RBM seeds: +1200 updates and frozen checks | Approximately 20–35 minutes |',
         '| Both RBM seeds: +3600 cumulative updates and frozen checks | Approximately 1–2 hours |',
         '| Original versus extended guides: all matched GFMC controls | Approximately 5–8 hours, if guides pass frozen checks |',
         '| Ten field/forward-time/population controls | Approximately 5–8 hours |',
         '| Gamma .55, both sizes/guides, all eight disorders and planned sensitivities | Approximately 8–12 hours |',
         '| Gamma .55–.60, both sizes, including sensitivities | Approximately 4–6 days |',
         '| All standard production training | Approximately 9–12 days |',
         '| Entire prescribed L4/L6 campaign, current two/two allocation | **Approximately 13–18 days (November 16–21)** |',
         '| Conditional: restore four trainers/six projectors after diagnostics | Approximately 5–7 days; requires an explicit scheduler change |', '',
         '**Measured production rates:**', '',
         '| Size / guide | Fits sampled | Median minutes per fit | Observed range |',
         '|---|---:|---:|---:|']
for row in rates:
    if row['phase'] == 'training':
        L, kind = row['group']
        lines.append(f"| L{L} / {kind} | {row['n']} | {row['median_seconds']/60:.1f} | {row['min_seconds']/60:.1f}–{row['max_seconds']/60:.1f} |")
lines += ['', '| Size / guide / GFMC setting | Runs sampled | Median minutes per run |', '|---|---:|---:|']
for row in rates:
    if row['phase'] == 'projection':
        L, kind, audit = row['group']
        lines.append(f"| L{L} / {kind} / {audit} | {row['n']} | {row['median_seconds']/60:.1f} |")
lines += ['', f"Remaining workload is **{work['training']:.1f} training worker-hours** and **{work['projection']:.1f} GFMC worker-hours**, calculated by multiplying each size/guide/audit median by its unfinished task count. Two workers give {work['training']/2:.1f} training wall-hours and {work['projection']/2:.1f} projection wall-hours. Training and projection overlap; these are not added. The idealized bottleneck is {baseline/24:.1f} days. Applying a stated 15–60% allowance for pipeline delays, guide-histogram sampling and throughput variation gives {baseline*1.15/24:.1f}–{baseline*1.60/24:.1f} days, rounded to 13–18 days.", '',
          'The extension checkpoints measured approximately 0.94 seconds per update per seed under the current split. Completed L6 fits indicate about eight additional minutes for frozen diagnostics per milestone. Extension GFMC starts after both training milestones; up to twelve runs share two CPU workers. Diagnostic and first-field milestones are planning judgments from these rates and queue sizes, not simulated schedules.', '',
          'The conditional four/six scenario has an idealized bottleneck of '+f'{restored/24:.1f} days'+', plus the diagnostic interval and throughput allowance. It is not the active configuration and was not enacted by this recalibration.', '',
          '**Limits:** Timing samples are predominantly Gamma=.55, with few disorders and changing concurrency. Higher fields, particularly .85/1.00, may change rates beyond this range. The 15–60% allowance is a planning scenario, not a statistical confidence interval. Imported L6 RBM pilot timings are excluded from the rate estimate, but their completed fits are credited. Active partial work is conservatively counted as unfinished. Projection timings omit subsequent guide-histogram sampling, covered by the allowance. Failed qualification can leave a prescribed context without valid results; no failed disorder is replaced. The schedule covers standard fits and the already launched finite diagnostics, not extending every RBM fit or additional convergence remedies. Scientific convergence and L8/L10 expansion have no defensible completion date yet.', '',
          '**Runtime policy:** no wall-clock cutoff; the finite authorized campaign continues.']
(dest/'TIMING.md').write_text('\n'.join(lines)+'\n')
shutil.copy2(__file__, archive/'recalibrate_campaign_time.py')
print(json.dumps({k:record[k] for k in ['snapshot_utc','completed_fits','completed_runs','remaining_worker_hours','current_planning_days']}, indent=2))
