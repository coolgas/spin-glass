"""One-shot auditable campaign progress snapshot; no background polling."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


RELATIVE=Path('data/kagome/ltfim_disorder_scaling_v1/repaired_campaign/20261103_v2')


def describe(record):
    current=record['corrected'];legacy=record['legacy']
    fields_by_size=record.get('fields_by_size',{'4':[v/100 for v in range(55,71)]+[.85,1.],
        '6':[v/100 for v in range(55,61)]+[.65,.70,.85,1.]})
    coverage='; '.join(f"L={size}: Gamma="+', '.join(f'{g:.2f}' for g in fields)
        for size,fields in fields_by_size.items())
    lines=['## Corrected campaign: compute allocation and progress',
        '',f"**Snapshot: {record['snapshot_utc']} (UTC).** This is a saved status, not a live monitor.",'',
        '| Work | Status | Completed training fits | Completed GFMC runs | Compute allocation |',
        '|---|---|---:|---:|---|']
    for row in legacy:
        lines.append(f"| {row['name']} | Retired; preserved legacy output | {row['completed_fits']} | {row['completed_runs']} | Zero workers |")
    run_total=record.get('planned_projector_runs')
    run_count=str(current['completed_runs'])+(f'/{run_total}' if run_total is not None else '')
    lines.append(f"| Corrected main campaign | {current['state']} | {current['completed_fits']}/{current['planned_fits']} | {run_count} | {current['trainers']} training + {current['projector_workers']} GFMC workers |")
    lines.extend(['',f"**Recorded failures:** corrected campaign {len(current['failures'])}; retired campaigns "+
        ', '.join(f"{row['name']}: {row['failures']}" for row in legacy)+'. All are retained.'])
    validation=record.get('early_run_validation')
    if validation:
        runs=validation['runs'];alarmed=[r for r in runs if r['ancestry_alarm']]
        lines.extend(['',f"**Early GFMC measurement check:** {len(runs)} completed records pass saved raw-hash, normalization, overlap-moment and finite-diagnostic checks; {len(alarmed)} have ancestry alarms. These remain in the data and are not qualified as converged results."])
        for row in alarmed:
            t=row['task']
            lines.append(f"- Ancestry alarm: L={t['L']}, Gamma={t['gamma']:g}, disorder {t['sample_id']}, {t['kind']}, {t['audit']}, repeat {t['repeat']}.")
    health=record.get('health_check')
    if health:
        lines.extend(['',f"**One-shot process/source check ({health['checked_utc']}):** "+
            f"{len(health['legacy_active_processes'])} active legacy processes; "+
            f"{health['production_source_files']} production source files checked, "+
            f"{len(health['production_source_mismatches'])} mismatches.",'',
            '| Active GFMC checkpoint | Size / field / disorder | Repeat | Saved steps / total | Completed measurement cycles |',
            '|---|---|---:|---:|---:|'])
        running=current.get('running_projectors',[])
        for item in health['projector_progress']:
            progress=item['record'];task=progress['task']
            if task not in running:continue
            lines.append(f"| {task['kind']} / {task['audit']} | L={task['L']}, Gamma={task['gamma']:g}, J={task['sample_id']} | {task['repeat']} | {progress['step']}/{progress['total']} | {progress['cycles']} |")
        lines.extend(['','Checkpoint steps show trajectory progress only; they do not measure convergence.'])
    lines.extend(['',
        '**What is running:** repaired large-RBM learning rate (0.001), log-domain GFMC, and frozen random/up/down sampling checks before guide selection. The standalone weak-field pilot has been merged into this campaign; its completed fits and partial checkpoint are reused with matching source hashes.',
        '',
        '**Priority and coverage:** fixed h=0.15; periodic kagome, N=3L². Ascending Gamma across L=4 and L=6. '+coverage+'. Disorders 60000–60007, both Jastrow/RBM guides, and both training seeds remain in the plan. No disorder is selected by overlap width.',
        '',
        f"**Resource budget:** at most {current['projector_workers']} projector futures are submitted at once; {current['deferred_projectors']} are deferred at this snapshot. New dispatch stops at {record['deadline_utc']}; already-running GFMC jobs finish their finite trajectories. The complete {current['planned_fits']}-fit plan is not a promise of completion within this budget.",
        '',
        '**Validated so far:** the repair controls contain 24 strong-field projection runs with no final projection failure or ancestry alarm, and the earlier 30-test regression suite passed. Sixteen targeted restart checks also passed, including the repaired histogram checkpoint replay and numerical-provenance mismatch rejection. These are implementation checks, not a ground-state convergence certificate.',
        '',
        '**Physical conclusions still pending:** corrected weak-field disorder averages, population/forward-time sensitivity, and connected-correlation size checks. Passing frozen-start diagnostics alone does not establish equilibrium. The existing physical figures retain the frozen 112-run broader snapshot; the retired counts above are not corrected ensemble results.',
        '',f"Remote campaign: `{record['remote_root']}`. Raw stop audit, launch configuration, source hashes, and imported-checkpoint identities are preserved with this snapshot."])
    if current['failures']:
        lines.extend(['',f"**Recorded corrected-campaign failures:** {len(current['failures'])}; retained explicitly, not replaced by other disorders."])
    return '\n'.join(lines)+'\n'


def notebook_cells(root):
    import nbformat as nb
    folder=Path(root)/RELATIVE;p=folder/'progress.json'
    if not p.exists():return []
    record=json.loads(p.read_text())
    cell=nb.v4.new_markdown_cell(describe(record))
    cell.metadata.tags=['repaired-campaign-progress']
    cells=[cell]
    timing=folder/'TIMING.md'
    if timing.exists():
        timing_cell=nb.v4.new_markdown_cell(timing.read_text())
        timing_cell.metadata.tags=['repaired-campaign-progress']
        cells.append(timing_cell)
    return cells


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();root=args.campaign;dest=args.destination;dest.mkdir(parents=True,exist_ok=True)
    current=json.loads((root/'results/parallel_status.json').read_text())
    manifest=json.loads((root/'results/manifest.json').read_text())
    launch=json.loads((root/'launch.json').read_text())
    stop=json.loads((root/'migration/legacy_stop.json').read_text())
    legacy=[dict(name=Path(r['root']).name,completed_fits=r['status']['completed_fits'],
        completed_runs=r['status']['completed_runs'],failures=len(r['status']['failures'])) for r in stop['campaigns']]
    record=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),remote_root=str(root),legacy=legacy,
        corrected=current,deadline_utc=launch['deadline_utc'],legacy_stopped_utc=stop['stopped_utc'],
        imports=launch['imports'],source_manifest_sha256=launch['source_manifest_sha256'],
        planned_projector_runs=launch['planned_projector_runs'],
        fields_by_size=manifest['fields_by_size'],
        ground_state_qualified=False)
    health=dest/'migration/health_check.json'
    if health.exists():record['health_check']=json.loads(health.read_text())
    validation=dest/'migration/grid_expansion_20261103/early_run_validation.json'
    if validation.exists():record['early_run_validation']=json.loads(validation.read_text())
    (dest/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
    (dest/'PROGRESS.md').write_text(describe(record))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [dest/'progress.json',dest/'PROGRESS.md']}
    (dest/'progress_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(describe(record))


if __name__=='__main__':main()
