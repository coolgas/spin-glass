"""Research-facing Stage B uncertainty, ancestry and provisional Stage C costing."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import numpy as np
from .experiment import write_json


OBSERVABLES=('energy_per_site','q2','chi_connected_per_site','q4','q_ea_finite')


def uncertainty(values):
    """Equal-weight disorder means; within-run variance is not added twice."""
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or a.shape[1]<2 or not len(a):raise ValueError('Need disorder x independent-repeat matrix')
    means=a.mean(axis=1);d,r=a.shape
    return dict(mean=float(means.mean()),disorder_se=float(means.std(ddof=1)/np.sqrt(d)) if d>1 else None,
        mc_se=float(np.sqrt(a.var(axis=1,ddof=1).sum()/r/d**2)),disorders=d,repeats=r)


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def assess(root):
    from .production_protocol import identity
    groups=defaultdict(list);expected=defaultdict(list);inputs={};all_records=[]
    for stage in ('notebook_stage_b','notebook_stage_b_escalation'):
        folder=root/stage
        if not (folder/'manifest.json').exists():continue
        manifest=json.loads((folder/'manifest.json').read_text())
        tasks=json.loads((folder/'tasks.json').read_text())
        assert identity({k:v for k,v in manifest.items() if k!='signature'})==manifest['signature']
        assert identity(tasks)==manifest['tasks_sha256']
        inputs[stage]=manifest['signature']
        for t in tasks:
            key=(t.get('ladder','base'),t['gamma'],t['h'],t['method'])
            expected[key].append(t)
            path=folder/'results'/(t['id']+'.json')
            if not path.exists():continue
            record=json.loads(path.read_text())
            assert record['task']==t and record['signature']==manifest['signature'] and record['state']=='completed'
            groups[key].append(record);all_records.append(record)
    rows=[];observables=[];cycle_rows=[];run_rows=[]
    for key,tasks in sorted(expected.items()):
        records=groups[key];level,g,h,method=key
        row=dict(level=level,gamma=g,h=h,method=method,completed=len(records),expected=len(tasks),
            alarms=sum(r['ancestry_alarm'] for r in records))
        times=[r['metrics']['seconds'] for r in records]
        row.update(median_seconds=float(np.median(times)) if times else None,
            p90_seconds=float(np.quantile(times,.9)) if times else None,
            peak_allocated_mib=max((r['metrics']['peak_allocated_mib'] for r in records),default=None),
            minimum_ess=min((r['minimum_descendant_ess'] for r in records),default=None))
        rows.append(row)
        by_disorder=defaultdict(dict)
        for record in records:
            t=record['task'];res=record['result'];metrics=record['metrics']
            by_disorder[t['sample_id']][t['repeat']]=record
            requested=metrics.get('requested_states');evaluated=metrics.get('evaluated_states');query=metrics.get('query_seconds')
            run_rows.append(dict(id=t['id'],level=level,gamma=g,h=h,method=method,sample_id=t['sample_id'],repeat=t['repeat'],
                seconds=metrics['seconds'],sample_seconds=metrics.get('sample_seconds'),query_seconds=query,
                requested_states=requested,evaluated_states=evaluated,
                evaluated_per_query_second=evaluated/query if query else None,
                cache_hit_fraction=1-evaluated/requested if requested else None,
                **{o:res[o] for o in OBSERVABLES}))
            for i,(ess,unique,lineage) in enumerate(zip(res['descendant_ess'],res['unique_ancestors'],res['largest_lineage'])):
                cycle_rows.append(dict(id=t['id'],level=level,gamma=g,h=h,method=method,sample_id=t['sample_id'],repeat=t['repeat'],
                    cycle=i,forward_time=res['forward_times'][i],ess=ess,unique_ancestors=unique,largest_lineage=lineage))
        full=[runs for runs in by_disorder.values() if set(runs)==set(range(4))]
        # Partial sets are listed in coverage; they cannot silently change the averaging weights.
        for obs in OBSERVABLES:
            stats=uncertainty([[runs[r]['result'][obs] for r in range(4)] for runs in full]) if full else dict(mean=None,disorder_se=None,mc_se=None,disorders=0,repeats=4)
            observables.append(dict(level=level,gamma=g,h=h,method=method,observable=obs,
                complete_context=len(records)==len(tasks),**stats))
    # One-disorder escalation contrasts use independent-run SE only. They do not
    # estimate variability across disorders or pass the Section 7 gate.
    comparisons=[]
    for g in (.5,2.):
        for h in (0.,.05):
            for method in ('network','constant'):
                baseline='base' if h==0 else 'baseline_h005'
                for low,high in ((baseline,'population256'),('population256','population1024'),
                                 ('population1024','equilibration4'),('population1024','forward2')):
                    a=[r for r in groups.get((low,g,h,method),[]) if r['task']['sample_id']==60000]
                    b=[r for r in groups.get((high,g,h,method),[]) if r['task']['sample_id']==60000]
                    if len(a)!=4 or len(b)!=4:continue
                    for obs in OBSERVABLES:
                        x=np.array([r['result'][obs] for r in a]);y=np.array([r['result'][obs] for r in b])
                        comparisons.append(dict(gamma=g,h=h,method=method,lower=low,higher=high,observable=obs,
                            difference=float(y.mean()-x.mean()),independent_run_se=float(np.sqrt(x.var(ddof=1)/4+y.var(ddof=1)/4)),
                            disorders=1,qualification_passed=False))
    # Stage C's nine initial contexts, eight disorders and four repeats. A
    # transparent linear-work scenario is provisional until W1024/time data exist.
    base=dict(W=512,eq=16.,measure=1.,fw=4.,cycles=16,branch=.1)
    configurations=[dict(name='candidate',**base)]
    for name,field,values in [('population','W',(2048,8192)),('equilibration','eq',(32.,64.)),
                              ('forward','fw',(8.,16.)),('cadence','branch',(.05,)),('cycles','cycles',(32,))]:
        for value in values:configurations.append(dict(name=name+str(value),**{**base,field:value}))
    configurations.extend([dict(name='alternative_start',**base),dict(name='combined_upper',W=8192,eq=64.,measure=1.,fw=16.,cycles=32,branch=.05)])
    def work(p):return p['W']*(p['eq']+p['measure']+p['cycles']*p['fw'])
    base_times=[r['median_seconds'] for r in rows if r['level']=='base' and r['method']=='network' and r['median_seconds'] is not None]
    costs=[]
    if base_times:
        for p in configurations:
            factor=work(p)/(64*7.)
            costs.append(dict(**p,runs=8*9*4,linear_gpu_hours=float(np.mean(base_times)*factor*8*9*4/3600),
                context_max_gpu_hours=float(max(base_times)*factor*8*9*4/3600)))
    budget=dict(state='provisional_linear_work_scenario',configurations=costs,
        gpu_hours_with_30_percent_contingency=1.3*sum(r['linear_gpu_hours'] for r in costs) if costs else None,
        assumptions='Serial GPU; baseline context medians equally weighted; linear W times total projection time; measurement time 1 is a costing assumption. Cadence, batching, cache and ancestry can violate scaling. Upper combined setting is a stress envelope, not a dispatch choice.',
        exclusions='Additional four-disorder screening, stress/remaining field contexts, extra repeats, equal-cost runs, further precision checks and final surveys.',
        measured_stage_c_budget=False)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_signatures=inputs,coverage=rows,
        observable_summary=observables,escalation_comparisons=comparisons,stage_c_cost=budget,
        gate_b_passed=False,stage_c_started=False,final_data_started=False,
        remaining_gate_b=['Complete the base and representative escalation comparisons.',
            'Validate cost scaling with W256/W1024 and longer times; record accepted GPU-hour/memory budget.',
            'Complete equal-end-to-end-cost comparison and cold/warm cache, event-count and timing-component profiling.'])
    destination=root/'stage_b_analysis';destination.mkdir(exist_ok=True)
    for name,data in [('coverage',rows),('observables',observables),('runs',run_rows),('ancestry_cycles',cycle_rows),('escalation_comparisons',comparisons),('stage_c_cost_scenario',costs)]:
        write_csv(destination/(name+'.csv'),data)
    write_json(destination/'summary.json',result)
    lines=['# Stage B research review','',result['created_utc'],
        '', '**Gate B: pending. Stage C and final-data collection have not started.**',
        '', 'All recorded ancestry alarms are retained. Coverage includes partial contexts; uncertainty tables use only disorders with all four independent repeats and label incomplete contexts. One-disorder escalation SE measures Monte Carlo variation only.',
        '', '| Setting | Gamma | h | Guide | Runs | Alarms | Median s | Worst ESS |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    def fmt(x):return 'pending' if x is None else f'{x:.3g}'
    for r in rows:
        lines.append(f"| {r['level']} | {r['gamma']} | {r['h']} | {r['method']} | {r['completed']}/{r['expected']} | {r['alarms']} | {fmt(r['median_seconds'])} | {fmt(r['minimum_ess'])} |")
    lines+=['','## Uncertainty and cost evidence','',
        'CSV exports in `stage_b_analysis/` retain every run and every ancestry cycle. `observables.csv` gives disorder SE and the separate MC contribution; these are not added together. Four cycles are not four independent repeats. Incomplete-disorder averages are descriptive and cannot qualify a context.',
        '', 'The Stage C costing worksheet explicitly enumerates 11 candidate/one-parameter/alternative-start/combined configurations, 288 guided runs each at the initial nine fields. Linear projection from available W64 timings gives **'+fmt(budget['gpu_hours_with_30_percent_contingency'])+' GPU-hours including 30% contingency**. This is an unvalidated scaling scenario, not an approved or measured budget. See `stage_c_cost_scenario.csv` for the configuration breakdown and context-maximum scenario.',
        '', budget['assumptions'], '', 'Excluded from that scenario: '+budget['exclusions'],
        '', 'Current records provide total execution-segment time, initialization sampling time, query time, requested/evaluated state counts and peak allocation. They do not isolate event counts, cold/warm cache phases, reporting time or propagation time. Those profiling requirements remain open.',
        '', 'Equal-settings guided/unguided outputs are available as coverage fills. An equal-cost experiment is still required; a runtime ratio alone is not an efficiency comparison.',
        '', '## Required next decisions','']
    lines+=['- '+item for item in result['remaining_gate_b']]
    lines+=['- Stage C: four fixed plus four runtime/ancestry-selected disorders; W512/2048/8192, equilibration16/32/64, forward4/8/16, cadence0.1/0.05, cycles16/32, alternative starts and combined checks. All 49 contexts require qualification before final admission.',
        '- Apply the same full anchor survey and acceptance criteria to L=4,6,8,10, with matched transition cuts at L=5,7,9.','']
    (root/'STAGE_B_REVIEW.md').write_text('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/kagome/transformer_af_production_v1'))
    args=parser.parse_args();result=assess(args.root)
    print(json.dumps({k:result[k] for k in ('created_utc','gate_b_passed','stage_c_started')}))


if __name__=='__main__':main()
