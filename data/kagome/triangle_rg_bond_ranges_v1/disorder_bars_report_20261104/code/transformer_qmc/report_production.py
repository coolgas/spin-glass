"""Audit mirrored execution artifacts; structural validity is not convergence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from .experiment import write_json
from .production_protocol import identity, seed_for


def audit_precision(root, signature):
    path=root/'precision_assessment/summary.json'
    if not path.exists():return dict(state='pending')
    summary=json.loads(path.read_text())
    assert summary['state']=='completed'
    assert not summary['production_precision_qualified'] and not summary['ground_state_qualified']
    assert hashlib.sha256((root/'assess_production_precision.py').read_bytes()).hexdigest()==summary['script_sha256']
    rows=summary['rows'];tested=[r for r in rows if r['tolerance'] is not None]
    assert len(rows)==245 and len(tested)==summary['comparisons']==147
    assert len({(r['gamma'],r['h'],r['observable']) for r in rows})==len(rows)
    assert sum(r['short_run_equivalence'] for r in tested)==summary['short_run_equivalence_passes']
    for row in rows:
        difference=np.asarray(row['per_disorder_precision_difference'])
        assert difference.shape==(8,) and np.isfinite(difference).all()
        np.testing.assert_allclose(difference.mean(),row['mean_precision_difference'],atol=1e-14)
        np.testing.assert_allclose(difference.std(ddof=1)/np.sqrt(8),row['disorder_paired_se'],atol=1e-14)
        bound=abs(row['mean_precision_difference'])+1.96*row['disorder_paired_se']
        np.testing.assert_allclose(bound,row['approx_95_abs_bound'],atol=1e-14)
        if row['tolerance'] is not None:assert row['short_run_equivalence']==(bound<=row['tolerance'])
    references=list((root/'precision_assessment').glob('L2_*.json'))
    assert len(references)==784
    for reference in references:
        value=json.loads(reference.read_text())
        assert value['source_signature']==signature and len(value['cycles'])==4
        for name,mean in value['means'].items():
            values=[cycle[name] for cycle in value['cycles']]
            assert np.isfinite(values).all()
            np.testing.assert_allclose(np.mean(values),mean,atol=1e-14)
    return {**{k:v for k,v in summary.items() if k!='rows'},
            'finite_forward_references':len(references),
            'short_run_equivalence_unresolved':len(tested)-summary['short_run_equivalence_passes']}


def audit_notebook_stage_b(root, original, stage='notebook_stage_b'):
    output=root/stage
    if not (output/'manifest.json').exists():return None
    manifest=json.loads((output/'manifest.json').read_text())
    assert identity({k:v for k,v in manifest.items() if k!='signature'})==manifest['signature']
    parent=original if stage=='notebook_stage_b' else json.loads((root/'notebook_stage_b/manifest.json').read_text())
    assert manifest['parent_signature']==parent['signature']
    assert manifest['source_sha256']==original['source_sha256']
    assert manifest['checkpoint_sha256']==original['checkpoint_sha256']
    script='notebook_stage_b.py' if stage=='notebook_stage_b' else 'research_followup.py'
    assert hashlib.sha256((root/script).read_bytes()).hexdigest()==manifest['script_sha256']
    if stage!='notebook_stage_b':
        assert hashlib.sha256((root/'notebook_stage_b_support.py').read_bytes()).hexdigest()==manifest['support_sha256']
    assert hashlib.sha256((root/'notebook_stage_b_protocol.ipynb').read_bytes()).hexdigest()==manifest['notebook_sha256']
    tasks=json.loads((output/'tasks.json').read_text())
    count=288 if stage=='notebook_stage_b' else 144
    assert identity(tasks)==manifest['tasks_sha256'] and len(tasks)==manifest['task_count']==count
    if stage=='notebook_stage_b':
        from .notebook_stage_b import build_tasks
    else:
        from .research_followup import build_tasks
    assert tasks==build_tasks(), 'Stage B differs from agreed notebook settings'
    expected={t['id']:t for t in tasks};counts=Counter();alarms=Counter();seconds={}
    for path in (output/'results').glob('*.json'):
        record=json.loads(path.read_text());task=record['task'];tid=task['id']
        assert path.stem==tid and expected[tid]==task and record['signature']==manifest['signature']
        assert record['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert record['state']=='completed' and not record['final_data'] and not record['ground_state_qualified']
        result=record['result'];parameters=task['parameters']
        for value in result.values():assert np.isfinite(np.asarray(value)).all(),tid
        for name in original['design']['observables']:
            assert name in result,(tid,name)
            if name in ('energy','energy_growth'):continue
            cycles=np.asarray(result[name+'_cycles']);assert cycles.shape[0]==4
            np.testing.assert_allclose(cycles.mean(0),result[name],atol=1e-12,rtol=1e-12)
        assert len(result['equilibration_trace'])==parameters['n_equilibrate'] and len(result['energy_trace'])==parameters['n_measure']
        assert len(result['growth_trace'])==parameters['n_measure'] and len(result['descendant_ess'])==4
        assert result['n_walkers']==parameters['n_walkers'] and result['n_replicas']==4 and result['tau_block']==.1
        np.testing.assert_array_equal(result['forward_times'],np.full(4,parameters['tau_forward']))
        assert record['ancestry_alarm']==bool(min(result['descendant_ess'])<8 or max(result['largest_lineage'])>.5)
        assert np.asarray(result['connected_spectrum']).shape==(16,)
        assert np.asarray(result['spin_sublattice_real']).shape==(16,3,3)
        np.testing.assert_allclose(result['connected_spectrum'][0],result['chi_connected'],atol=1e-10)
        method=task['method'];counts[method]+=1;alarms[method]+=int(record['ancestry_alarm'])
        seconds.setdefault(method,[]).append(record['metrics']['seconds'])
    status_path=output/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'initialized'}
    return dict(signature=manifest['signature'],completed=sum(counts.values()),expected=count,
        completed_by_method=dict(counts),ancestry_alarms_by_method=dict(alarms),supervisor=status,
        specification=manifest.get('specification',manifest.get('selection')),deadline_unix=manifest['deadline_unix'],
        median_seconds_by_method={k:float(np.median(v)) for k,v in seconds.items()},
        active_failures=[p.name for p in (output/'failures').glob('*.json')],
        provenance_and_structure_passed=True,final_data_started=False,ground_state_qualified=False)


def audit(root):
    pipeline=root/'pipeline'
    manifest=json.loads((pipeline/'manifest.json').read_text());signature=manifest['signature']
    assert identity({k:v for k,v in manifest.items() if k!='signature'})==signature
    tasks=json.loads((pipeline/'tasks.json').read_text())
    assert identity(tasks)==manifest['tasks_sha256'] and len(tasks)==manifest['task_count']
    expected={t['id']:t for t in tasks};assert len(expected)==len(tasks)
    for task in tasks:
        no_id={k:v for k,v in task.items() if k!='id'}
        assert identity(no_id)[:24]==task['id']
        assert seed_for({k:v for k,v in no_id.items() if k!='seed'})==task['seed']
    for folder,meta in [('runner_source',manifest)]+[('qualification_source',json.loads((root/'qualification'/('manifest_'+stage+'.json')).read_text())) for stage in ('gpu','exact')]:
        for name,digest in meta['source_sha256'].items():
            assert hashlib.sha256((root/folder/'code'/name).read_bytes()).hexdigest()==digest,name
    counts=Counter();stages=Counter();alarms=Counter();methods=Counter();network=Counter();constant=Counter();latest=0.;completed=0
    for path in (pipeline/'results').glob('*.json'):
        record=json.loads(path.read_text());task=record['task'];tid=task['id']
        assert path.stem==tid and expected[tid]==task and record['signature']==signature
        assert record['state']=='completed' and not record['final_data'] and not record['ground_state_qualified']
        result=record['result'];L=task['L'];cycles=task['parameters']['n_cycles']
        for value in result.values():assert np.isfinite(np.asarray(value)).all(),tid
        for name in manifest['design']['observables']:
            assert name in result,(tid,name)
            if name in ('energy','energy_growth'):continue
            arr=np.asarray(result[name+'_cycles']);assert arr.shape[0]==cycles,(tid,name,arr.shape)
            np.testing.assert_allclose(arr.mean(0),result[name],atol=1e-12,rtol=1e-12)
        assert np.asarray(result['connected_spectrum']).shape==(L*L,)
        assert np.asarray(result['spin_sublattice_real']).shape==(L*L,3,3)
        np.testing.assert_allclose(result['connected_spectrum'][0],result['chi_connected'],atol=1e-10)
        assert len(result['descendant_ess'])==cycles
        counts[L]+=1;stages[task['stage']]+=1;alarms[L]+=int(record['ancestry_alarm'])
        methods[task['method']]+=1
        if task['method']=='network':network[L]+=1
        elif task['method']=='constant':constant[L]+=1
        latest=max(latest,record['finished_unix']);completed+=1
    direct=list((root/'qualification/direct_checks').glob('*.json'))
    exact=list((root/'qualification/exact').glob('*.json'));max_rate=0.;max_residual=0.
    for path in direct+exact:
        record=json.loads(path.read_text())
        assert record['status']=='completed' and record['signature']==manifest['checks_signature']
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==record['npz_sha256']
        max_rate=max(max_rate,record.get('max_relative_rate_error',0))
        max_residual=max(max_residual,record.get('residual',0))
    def load(name):
        p=pipeline/name
        return json.loads(p.read_text()) if p.exists() else {}
    gpu=[];telemetry=pipeline/'gpu.csv'
    if telemetry.exists():
        for line in telemetry.read_text().splitlines():
            try:gpu.append(float(line.split(',')[1]))
            except (ValueError,IndexError):pass
    return dict(provenance_and_structure_passed=True,signature=signature,completed=completed,expected=len(tasks),
        completed_by_size=dict(counts),completed_by_stage=dict(stages),ancestry_alarms_by_size=dict(alarms),
        completed_by_method=dict(methods),network_completed_by_size=dict(network),constant_completed_by_size=dict(constant),
        direct_checks=len(direct),exact_references=len(exact),max_pointwise_relative_rate_difference=max_rate,
        max_exact_residual=max_residual,latest_result_unix=latest,supervisor=load('status.json'),throughput=load('throughput.json'),
        final_data_started=False,notebook_stage_b=audit_notebook_stage_b(root,manifest),
        notebook_stage_b_escalation=audit_notebook_stage_b(root,manifest,'notebook_stage_b_escalation'),
        precision_assessment=audit_precision(root,manifest['checks_signature']),
        active_failures=[p.name for p in (pipeline/'failures').glob('*.json')],
        reviewed_failures=[str(p.relative_to(pipeline)) for p in (pipeline/'reviewed_failures').rglob('review.json')],
        gpu_recent_mean_percent=float(np.mean(gpu[-30:])) if gpu else None,audited_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/kagome/transformer_af_production_v1'))
    args=parser.parse_args();s=audit(args.root);write_json(args.root/'execution_audit.json',s)
    if s['notebook_stage_b']:
        from .assess_stage_b import assess
        assess(args.root)
    stage=s['notebook_stage_b']
    escalation=s['notebook_stage_b_escalation']
    pointer=args.root/'active_stage.json'
    active_name=json.loads(pointer.read_text())['stage'] if pointer.exists() else 'preliminary'
    selected=s.get(active_name) or stage
    active=selected['supervisor'] if selected else s['supervisor']
    lines=['# Production pipeline execution status','','Snapshot audited at '+s['audited_utc']+'.',
           '', '**State:** '+active.get('state','not yet recorded'),
           '', '**Scientific status:** qualification-stage artifacts only. Final-data collection and ground-state qualification are pending.',
           '', 'Anchor sizes L=4,6,8,10 retain all 49 contexts, initially 64 fresh disorders, four repeats and the same 16-disorder unguided comparison subset. L=5,7,9 retain matched transition cuts.']
    if stage:
        lines+=['','## Active: notebook Section 5, Stage B','',
                'The specified L10 task completed. The earlier tiny-probe queue is paused and superseded; it does not count toward Stage B.',
                '', 'L=4 (48 spins), PBC, W=64 four-replica groups; equilibration 2, energy-measurement time 1, forward time 1, four cycles and branch interval 0.1. Both methods use float64.',
                '', 'Four development disorders, Gamma=0.5,1,2, h=0,0.15,0.3, and four repeats: 288 population runs.',
                '', '| Method | Completed | Expected | Ancestry alarms |','|---|---:|---:|---:|']
        for method in ('network','constant'):
            lines.append('| '+method+' | '+str(stage['completed_by_method'].get(method,0))+' | 144 | '+str(stage['ancestry_alarms_by_method'].get(method,0))+' |')
        lines+=['','Stage B active failures: '+str(len(stage['active_failures']))+'. Structural checks pass; this does not certify convergence.',
                '', 'Next notebook steps: assess runtime/ancestry, escalate representative cases to W=256 and 1024 and longer times (including h=0.05), then evaluate cost feasibility before Stage C. Stage C and final-disorder collection are not auto-promoted.',
                ]
    if escalation:
        lines+=['','## Section 5 escalation schedule','',
            'State: '+escalation['supervisor'].get('state','initialized')+'. Completed '+str(escalation['completed'])+'/144.',
            '', 'Fixed disorder 60000; Gamma=0.5,2; h=0,0.05; both guides, float64 and four repeats. W64 small-field baseline, W256/W1024, then separate equilibration 4 and forward time 2 at W1024. Measurement 1, four cycles and branch interval 0.1 throughout.',
            '', 'The controller requires the full base pilot and holds the shared supervisor lock. The active status records its operational deadline; separately documented continuations preserve the original launch and frozen scientific identities. This is representative feasibility evidence; Stage C still requires eight disorders and substantive convergence ladders.',
            '', 'Scientific assessments: [STAGE_B_REVIEW.md](STAGE_B_REVIEW.md) and [PHYSICAL_CONVERGENCE.md](PHYSICAL_CONVERGENCE.md).']
        deadline=escalation['supervisor'].get('deadline_unix')
        if deadline:
            lines+=['','Active escalation deadline UTC: '+datetime.fromtimestamp(deadline,timezone.utc).isoformat()+'.']
    lines+=['','## Archived preliminary queue (paused)']
    lines+=[
           '', '| Item | Completed | Expected |','|---|---:|---:|',
           '| Direct-guide checks | '+str(s['direct_checks'])+' | 392 |',
           '| Exact reference contexts | '+str(s['exact_references'])+' | 392 |',
           '| Qualification/feasibility runs | '+str(s['completed'])+' | '+str(s['expected'])+' |',
           '', '| L | Completed runs | Direct network | Unguided | Ancestry alarms |','|---:|---:|---:|---:|---:|']
    for L in [2,4,5,6,7,8,9,10]:
        lines.append('| '+str(L)+' | '+str(s['completed_by_size'].get(L,0))+' | '+str(s['network_completed_by_size'].get(L,0))+' | '+str(s['constant_completed_by_size'].get(L,0))+' | '+str(s['ancestry_alarms_by_size'].get(L,0))+' |')
    lines+=['','L=2 uses the cached direct-rate precision ensemble. Larger-size CPU and GPU lanes progress at different rates; total completion does not imply the direct-network lane is complete.',
            '', 'All alarm cases are retained. Completed short runs are not certified final measurements.',
            '', '## Throughput and recovery','',
            'Selected GPU workers: '+str(s['throughput'].get('selected_workers','pending'))+'. CPU workers: '+str(s['supervisor'].get('cpu_workers','pending'))+'.',
            '', '| GPU workers | Matched 8-task wave seconds |','|---:|---:|']
    for row in s['throughput'].get('results',[]):
        lines.append('| '+str(row['workers'])+' | '+format(row['wall_seconds'],'.2f')+' |')
    precision=s['precision_assessment']
    if precision['state']=='completed':
        lines+=['','## Short-setting precision assessment','',
                str(precision['finite_forward_references'])+' finite-forward reference contexts completed for 49 fields, eight disorders and two precisions.',
                '', str(precision['short_run_equivalence_passes'])+' of '+str(precision['comparisons'])+' approximate paired-disorder 95% bounds meet the declared tolerances; '+str(precision['short_run_equivalence_unresolved'])+' remain unresolved.',
                '', 'These are comparisons at one short setting with four independent repeats, not simultaneous confidence guarantees. Unresolved bounds do not establish a float32 bias. Float64 remains the comparison implementation; neither long-time float32 accuracy nor ground-state convergence is qualified.']
    lines+=['','GPU mean over the latest up-to-30 telemetry samples: '+str(s['gpu_recent_mean_percent'])+'%.',
            '', 'Each bounded process window checkpoints and pauses workers at expiry. The original 12-hour launch record and separate continuation windows are retained.',
            '', 'Source hashes, task identities, finite outputs, cycle means and spectral normalization pass the structural audit. Precision equivalence and projection convergence remain separate gates.',
            '', 'Active failures: '+str(len(s['active_failures']))+'. Reviewed incidents: '+str(len(s['reviewed_failures']))+'.',
            '', 'Larger-size development IDs: 60000:60032. Final IDs 61000:61256 remain unused.',
            '', 'Next: ensemble precision, population/time/cadence convergence, alternative starts, crossing-error targets and a measured final-campaign budget. This qualification queue is separate from the 62,720-run final anchor survey.','']
    (args.root/'STATUS.md').write_text(chr(10).join(lines))
    print(json.dumps(dict(archived_probe_completed=s['completed'],
        active_stage=active_name,
        active_completed=selected['completed'] if selected else s['completed'],
        active_expected=selected['expected'] if selected else s['expected'],
        active_failures=selected['active_failures'] if selected else s['active_failures'],
        provenance_and_structure_passed=True)))


if __name__=='__main__':
    main()
