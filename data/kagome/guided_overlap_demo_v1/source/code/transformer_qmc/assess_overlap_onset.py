"""Exact two-copy overlap distributions and field-resolved ancestry diagnostics."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def walsh(values):
    result=np.asarray(values,dtype=float).copy()
    n=result.size
    if result.ndim!=1 or n<2 or n&(n-1):raise ValueError('Need a power-of-two vector')
    half=1
    while half<n:
        blocks=result.reshape(-1,2*half)
        left=blocks[:,:half].copy();right=blocks[:,half:].copy()
        blocks[:,:half]=left+right;blocks[:,half:]=left-right
        half*=2
    return result


def overlap_distribution(probability):
    """Exact P(q) via XOR convolution of two independent z-basis draws."""
    p=np.asarray(probability,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:
        raise ValueError('Invalid probability')
    n=int(round(math.log2(len(p))))
    if len(p)!=2**n or n<1:raise ValueError('Need full computational basis')
    p=p/p.sum()
    xor=walsh(walsh(p)**2)/len(p)
    if xor.min() < -1e-12:raise FloatingPointError('Negative XOR probability')
    xor=np.maximum(xor,0)
    distances=np.fromiter((bin(i).count('1') for i in range(len(p))),dtype=int)
    mass=np.bincount(distances,weights=xor,minlength=n+1)
    mass/=mass.sum()
    return (1-2*np.arange(n+1)/n)[::-1],mass[::-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/kagome/transformer_af_production_v1'))
    args=parser.parse_args();root=args.root;out=root/'overlap_onset';out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    def csvfile(name,rows):
        if rows:
            with (out/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    hashes={};rows=[];groups=defaultdict(list);tv=[]
    manifest=json.loads((root/'pipeline/manifest.json').read_text())
    for path in sorted((root/'qualification/exact').glob('*.json')):
        raw=path.read_bytes();meta=json.loads(raw);npz=path.with_suffix('.npz');binary=npz.read_bytes()
        assert meta['status']=='completed' and meta['signature']==manifest['checks_signature']
        assert hashlib.sha256(binary).hexdigest()==meta['npz_sha256']
        assert meta['L']==2 and meta['residual']<=1e-9
        with np.load(npz) as data:p=data['amplitude']**2
        np.testing.assert_allclose(p.sum(),1,atol=1e-10,rtol=0)
        q,mass=overlap_distribution(p)
        for power,name in [(1,'q'),(2,'q2'),(4,'q4')]:
            np.testing.assert_allclose(mass@q**power,meta['observables'][name],atol=2e-11,rtol=0)
        if meta['h']==0:np.testing.assert_allclose(mass,mass[::-1],atol=1e-10,rtol=0)
        hashes[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
        hashes[str(npz.relative_to(root))]=hashlib.sha256(binary).hexdigest()
        distributions={'exact':mass}
        guide=root/'qualification/direct_checks'/path.name
        if guide.exists():
            gmeta=json.loads(guide.read_text());gnpz=guide.with_suffix('.npz');gbinary=gnpz.read_bytes()
            assert gmeta['signature']==meta['signature'] and gmeta['status']=='completed'
            assert hashlib.sha256(gbinary).hexdigest()==gmeta['npz_sha256']
            with np.load(gnpz) as data:
                logs=data['logs64'];gp=np.exp(2*(logs-logs.max()))
            _,distributions['guide']=overlap_distribution(gp)
            hashes[str(guide.relative_to(root))]=hashlib.sha256(guide.read_bytes()).hexdigest()
            hashes[str(gnpz.relative_to(root))]=hashlib.sha256(gbinary).hexdigest()
            tv.append(dict(sample_id=meta['sample_id'],gamma=meta['gamma'],h=meta['h'],
                total_variation=.5*float(abs(mass-distributions['guide']).sum())))
        for method,distribution in distributions.items():
            groups[(method,meta['gamma'],meta['h'])].append(distribution)
            for value,weight in zip(q,distribution):
                rows.append(dict(L=2,N=12,sample_id=meta['sample_id'],gamma=meta['gamma'],h=meta['h'],method=method,q=float(value),probability=float(weight)))
    if not rows:raise RuntimeError('No exact artifacts found')
    summary=[]
    for (method,g,h),values in sorted(groups.items()):
        a=np.asarray(values);mean=a.mean(0);se=a.std(0,ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.full(len(q),np.nan)
        for x,y,e in zip(q,mean,se):summary.append(dict(method=method,gamma=g,h=h,disorders=len(a),q=float(x),mean_probability=float(y),disorder_se=float(e)))
    csvfile('exact_and_guide_pq_by_disorder.csv',rows);csvfile('pq_disorder_summary.csv',summary);csvfile('guide_exact_total_variation.csv',tv)
    # Keep h, numerical setting and guide separate. Cycles are not trials.
    counts=defaultdict(list);expected=defaultdict(list)
    for stage in ('notebook_stage_b','notebook_stage_b_escalation'):
        folder=root/stage
        if not (folder/'manifest.json').exists():continue
        stage_manifest=json.loads((folder/'manifest.json').read_text())
        tasks=json.loads((folder/'tasks.json').read_text())
        for t in tasks:
            key=(stage,t.get('ladder','base'),t['gamma'],t['h'],t['method'],t['sample_id'])
            expected[key].append(t);path=folder/'results'/(t['id']+'.json')
            if not path.exists():continue
            raw=path.read_bytes();r=json.loads(raw)
            assert r['task']==t and r['signature']==stage_manifest['signature'] and r['state']=='completed'
            hashes[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
            counts[key].append(r)
    ancestry=[]
    for key,tasks in sorted(expected.items()):
        stage,level,g,h,method,sid=key;records=counts[key];p=tasks[0]['parameters']
        ancestry.append(dict(stage=stage,level=level,gamma=g,h=h,method=method,sample_id=sid,
            W=p['n_walkers'],tau_eq=p['n_equilibrate']*p['tau_block'],tau_fw=p['tau_forward'],
            completed=len(records),expected=len(tasks),alarms=sum(r['ancestry_alarm'] for r in records),
            complete_repeats=len(records)==len(tasks),
            median_min_ess_fraction=float(np.median([r['minimum_descendant_ess']/p['n_walkers'] for r in records])) if records else None))
    csvfile('ancestry_by_disorder.csv',ancestry)
    mismatch=defaultdict(list)
    for item in tv:mismatch[(item['gamma'],item['h'])].append(item['total_variation'])
    mismatch_lines=['| Gamma | h=0 | h=0.05 | h=0.15 | h=0.30 |',
                    '|---:|---:|---:|---:|---:|']
    for g in (.25,.5,.75,1.,1.5,2.,3.):
        mismatch_lines.append('| '+str(g)+' | '+' | '.join(
            f'{np.mean(mismatch[(g,h)]):.5f}' for h in (0.,.05,.15,.3))+' |')
    # A planning artifact, deliberately separate from the dispatch queues.
    proposal=dict(schema=1,state='proposed_pending_escalation_cost_review',dispatched=False,
        L=4,N=48,periodic=True,method='network',dtype='float64',
        disorders=list(range(60000,60004)),repeats=list(range(4)),
        gamma=[.5,.75,1.,1.25,1.5,1.75,2.],h=[0.,.05],walkers=[256,1024],
        equilibration=2.,measurement=1.,forward=1.,cycles=4,branch=.1,
        planned_combinations=448,
        prerequisites=['Completed population/time escalation assessment',
            'Measured runtime and remaining process budget',
            'Freeze executable tasks, source hashes and independent seeds before dispatch',
            'Reuse prior results only when full numerical and RNG identity agrees'],
        physical_refinement=dict(anchor_sizes=[4,6,8,10],additional_cut_sizes=[5,7,9],
            fields='Common midpoints of measured physical crossing brackets; presently pending',
            admission='Population/time/cadence qualification and measured feasible cost at every size',
            observables=['Binder ratio','connected correlation ratio'],
            confirmation='Fresh disorders after development bracket selection',
            extrapolation='Joint bootstrap crossings; fixed drift powers 1,2,3; size-window sensitivity'),
        ground_state_qualified=False)
    (out/'next_stage_proposal.json').write_text(json.dumps(proposal,indent=2))
    gammas=sorted({r['gamma'] for r in rows});fields=(0.,.05,.15,.3)
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True,constrained_layout=True)
    vmax=max(float(np.mean(v,axis=0).max()) for v in groups.values())
    for row,method in enumerate(('exact','guide')):
        for col,h in enumerate(fields):
            image=np.array([np.mean(groups[(method,g,h)],axis=0) for g in gammas])
            im=axes[row,col].imshow(image,origin='lower',aspect='auto',vmin=0,vmax=vmax,cmap='magma')
            axes[row,col].set(xticks=[0,len(q)//2,len(q)-1],xticklabels=['-1','0','1'],
                yticks=range(len(gammas)),yticklabels=gammas,xlabel='Overlap q',ylabel='Gamma / K',title=f'{method}, h={h:g}')
    fig.colorbar(im,ax=axes,label='Probability mass per q bin',shrink=.8)
    fig.suptitle('12-spin PBC: disorder-averaged two-copy P(q); guide before projection')
    for ext in ('png','pdf'):fig.savefig(out/('exact_guide_pq.'+ext))
    plt.close(fig)
    now=datetime.now(timezone.utc).isoformat()
    lines=['# Ancestry onset and exact overlap-distribution study','',now,'',
        '## Scientific question','',
        'Does deterioration of guided-GFMC ancestry as Gamma decreases coincide with changes in physical overlap distributions? Treat this as a hypothesis to test. Ancestry depends on the guide, W, projection time, branch cadence, number of cycles and disorder; it is not an order parameter.',
        '', '## Exact definition and completed calculation','',
        'For two independent z-basis measurements of the same disorder ground state, q = sum_i s_i t_i / N and P_J(q) = sum_(s,t) |psi_J(s)|^2 |psi_J(t)|^2 1[q(s,t)=q]. This is an equal-time measurement distribution, not a decomposition into thermodynamic pure states.',
        '', f"Computed {len(rows)//13} distributions (exact plus available guide distributions) from existing 12-spin (L=2) references. No new diagonalization was needed. Each exact distribution is checked against stored q, q2 and q4, normalization and h=0 spin-flip symmetry. The guide distribution uses the frozen float64 guide before GFMC projection.",
        '', '![Exact and guide overlap distributions](exact_guide_pq.png)',
        '', 'All 49 field contexts and eight disorders are exported. Discrete q bins have spacing 2/12. Disorder-average error bars are descriptive SE across disorders; exact diagonalization itself introduces no Monte Carlo error. At h=0 the finite-system P(q) is symmetric; nonzero h can shift and skew it. A 12-spin system has finite-size crossovers, not evidence of a thermodynamic singularity.',
        '', '## Measured guide mismatch','',
        'The table gives the disorder mean of total variation distance, TV = (1/2) sum_q |P_exact(q)-P_guide(q)|. Zero means identical distributions; one means disjoint support. These compare the exact ground state with the unprojected guide, not with a converged GFMC estimate.',
        '', *mismatch_lines, '',
        'The guide agrees closely at Gamma >= 1.5, while the mismatch increases at lower Gamma. At h=0 the exact distribution moves toward both overlap extremes; nonzero h favors positive overlap. The guide underrepresents this concentration. This makes guide error a concrete possible contributor to ancestry loss. These 12-spin data cannot establish how much of the 48-spin ancestry loss is caused by guide error or identify a phase boundary.',
        '', '## Ancestry onset protocol','',
        '1. Finish the active base and population/time tests. Report alarm fractions separately for each h and disorder, together with ESS/W, ancestor counts and largest lineage. The existing alarm is ESS < 8 or largest lineage > 0.5 in any cycle. Keep the cycle count fixed for onset comparisons.',
        '2. At h=0 and h=0.05, scan downward from Gamma=2 through 1.75, 1.5, 1.25, 1 and 0.75. Retain Gamma=0.5 as the difficult endpoint. First use the same four fixed development disorders, four repeats, W256 and W1024, equilibration2, measurement1, forward1, cycles4, branch0.1 and float64 guide. New settings remain feasibility diagnostics. This scan is proposed, not dispatched; cost review follows the current escalations.',
        '3. Summarize a diagnostic half-alarm crossing only when complete data bracket alarm fraction 0.5. Interpolate between adjacent tested Gamma values and show per-disorder curves; report nonmonotonicity or multiple crossings instead of forcing a threshold. Refine bracket midpoints only after measuring cost. Confidence assessment must respect disorder clustering; four disorders give limited precision.',
        '4. Test whether the onset moves with W, forward time and guide choice. A moving onset identifies an algorithmic limitation; a stable onset is a candidate region for physical investigation, not proof of a transition.',
        '5. Compare exact P(q), q2, q4 and connected correlations across the same fields. Compare the frozen guide with exact P(q) on each identical 12-spin disorder to diagnose guide mismatch. The current 12-spin and 48-spin disorder pools are different; cross-size comparisons are ensemble comparisons, not paired-sample tests.',
        '6. Add descendant-weighted P(q) histogram tags to a separately validated projector snapshot. Existing scalar q-cycle averages cannot reconstruct P(q). Verify histogram normalization, moments and exact finite-forward targets at L=2 before using it at L=4. Complete convergence and finite-size checks before estimating a physical boundary.',
        '', '## Refinement and finite-size extrapolation','',
        'Use the ancestry bracket to prioritize fields. At each fixed h, obtain qualified Binder and connected-correlation ratios on common field cuts for L=4,6,8,10, then L=5,7,9 with compatible geometry. The 12-spin exact result is a benchmark outside the primary scaling fit. Refine measured physical crossing brackets with common midpoint fields and fresh confirmation disorders. Retain absent or multiple crossings and their bootstrap frequency; do not silently condition the fit on successful crossing samples.',
        '', 'Resample each disorder whole across its field curve and moment set, and independently across sizes. Recompute every crossing jointly so shared-size covariance survives. Conditional fits use Gamma_cross = Gamma_c + A (sqrt(L1 L2))^(-lambda), initially checking fixed lambda=1,2,3. This effective-length ansatz is phenomenological; use compatible pair families, preferably fixed size ratios when available. Vary the minimum size and pair family and report statistical, interpolation, numerical and model sensitivity separately.',
        '', 'The tested utility `finite_size_crossings.py` locates crossings within measured brackets and fits joint bootstrap crossing samples with their covariance. At least three distinct effective lengths are required. Synthetic recovery tests validate the computation, not its physical applicability. Scaling collapse R_L = F((Gamma-Gamma_c)L^(1/nu)) is conditional on evidence for a continuous transition and suitable scaling of the chosen ratio. Sparse sizes may leave both Gamma_c and nu unresolved.',
        '', '**Current inference status:** no qualified multi-size physical curves are available, so no measured physical crossing, thermodynamic transition estimate or critical exponent is reported. `next_stage_proposal.json` preserves the 448-combination guided onset design and common-size refinement requirements; it is not a dispatched queue. Cost and qualification evidence from the running escalation determine the next executable batch.',
        '', '## Files','',
        '- `exact_and_guide_pq_by_disorder.csv`: every P_J(q) bin.',
        '- `pq_disorder_summary.csv`: equal-weight disorder means and SE.',
        '- `guide_exact_total_variation.csv`: per-context/disorder guide mismatch.',
        '- `ancestry_by_disorder.csv`: field-resolved coverage and alarms, retaining incomplete groups.',
        '- `next_stage_proposal.json`: proposed onset settings and gated finite-size refinement.',
        '- `provenance.json`: input hashes and numerical/scientific limits.','']
    (out/'RESEARCH_NOTE.md').write_text('\n'.join(lines))
    (out/'provenance.json').write_text(json.dumps(dict(created_utc=now,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashes,exact_contexts=len(rows)//13-len(tv),guide_contexts=len(tv),phase_transition_identified=False,
        additional_gamma_scan_dispatched=False),indent=2))
    print(out/'RESEARCH_NOTE.md')


if __name__=='__main__':main()
