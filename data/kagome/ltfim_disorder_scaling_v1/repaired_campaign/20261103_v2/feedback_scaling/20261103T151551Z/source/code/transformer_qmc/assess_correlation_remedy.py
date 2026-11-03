"""Integrity checks and conservative fixed-disorder assessment of the L2 remedy."""
import argparse
import json
from pathlib import Path

import numpy as np

from .assess_correlation_escalation import bootstrap, comparison
from .correlation_lengths import ROOT, length, sha
from .correlation_remedy_campaign import FIELDS, IDS, SETTINGS, parameters, write
from .extend_correlation_fields import stem, load_inputs
from .independent_replica_projector import spectrum_from_moments
from .production_protocol import identity, geometry


def finite(value):
    return float(value) if np.isfinite(value) else None


def disorder_mc_se(raw):
    rng=np.random.default_rng(20261102)
    d,r,_=raw.shape
    disorders=rng.integers(d,size=(4000,d))
    repeats=rng.integers(r,size=(4000,d,r))
    draws=length(raw[disorders[...,None],repeats].mean((1,2)),2)
    return finite(np.std(draws[np.isfinite(draws)],ddof=1))


def bound_check(mean, boot, target, other=None):
    check = comparison(mean, boot, target, other)
    spread = boot.std(0,ddof=1) if other is None else (boot-other).std(0,ddof=1)
    return check, spread


def within(check, spectral):
    return bool(check['xi_over_side_95_bound'] is not None and
                check['xi_over_side_95_bound'] <= .01 and
                check['invalid_bootstrap_fraction'] == 0 and np.all(spectral <= .05))


def assess(root):
    root=Path(root); mp=root/'manifest.json'; manifest=json.loads(mp.read_text())
    signature=identity(manifest); lattice=geometry(2)
    assert manifest['gamma']==FIELDS and manifest['sample_ids']==IDS and manifest['settings']==SETTINGS
    assert manifest['exact_labels_used'] is False
    for relative,digest in manifest['source_sha256'].items():
        assert sha(ROOT/relative)==digest,relative
    guides={}; records={}; input_hashes={str(mp):sha(mp)}
    for gamma in FIELDS:
        for sid in IDS:
            path=root/'guides'/(stem(gamma,sid)+'.json')
            if not path.exists(): continue
            record=json.loads(path.read_text()); _,_,inputs,checkpoint=load_inputs(gamma,sid)
            assert record['signature']==signature and record['status']=='completed'
            assert (record['gamma'],record['sample_id'],record['L'],record['h'])==(gamma,sid,2,.15)
            assert record['input_sha256']==inputs and record['checkpoint_sha256']==checkpoint
            assert record['exact_labels_used'] is False and record['degree']==4
            assert sha(path.with_suffix('.npz'))==record['npz_sha256']
            with np.load(path.with_suffix('.npz')) as data:
                assert data['logs'].shape==(4096,) and np.isfinite(data['logs']).all()
                np.testing.assert_allclose(np.exp(2*data['logs']).sum(),1.,atol=1e-12)
            guides[gamma,sid]=record
            input_hashes[str(path)]=sha(path);input_hashes[str(path.with_suffix('.npz'))]=sha(path.with_suffix('.npz'))
    expected={(g,s,k,r) for g in FIELDS for s in IDS for k in SETTINGS for r in range(4)}
    expected|={(g,s,k,r) for g,s in [(.25,50004),(.5,50003),(.75,50005)]
               for k in ['original_independent','corrected_joint'] for r in range(4)}
    for path in sorted((root/'runs').glob('*.json')):
        record=json.loads(path.read_text());g,s,k,r=[record[x] for x in ['gamma','sample_id','setting','repeat']]
        key=(g,s,k,r);assert key in expected and key not in records,key
        assert path.name==stem(g,s)+f'_{k}_r{r}.json'
        assert record['signature']==signature and record['status']=='completed'
        assert record['L']==2 and record['h']==.15 and record['ground_state_qualified'] is False
        assert record['seed']==int(identity(['correlation-remedy-v1',g,s,k,r])[:16],16)
        assert record['parameters']==parameters(k if k in SETTINGS else 'base')
        gm=guides[g,s]
        assert record['input_sha256']==gm['input_sha256'] and record['checkpoint_sha256']==gm['checkpoint_sha256']
        assert record['guide_sha256']==(None if k=='original_independent' else gm['npz_sha256'])
        result=record['result']; spectrum=np.asarray(result['connected_spectrum'])
        assert spectrum.shape==(4,) and np.isfinite(spectrum).all()
        if k!='corrected_joint':
            m=np.asarray(result['magnetization_cycles']);q=np.asarray(result['pair_cycles'])
            assert m.shape==(4,16,12) and q.shape==(4,16,12,12)
            assert np.isfinite(m).all() and np.isfinite(q).all()
            np.testing.assert_allclose(spectrum,spectrum_from_moments(m.mean(1),q.mean(1),lattice),atol=1e-12)
            record['halves']=[spectrum_from_moments(m[:,part].mean(1),q[:,part].mean(1),lattice) for part in [slice(0,8),slice(8,16)]]
            assert result['seed']==record['seed'] and result['n_walkers']==record['parameters']['n_walkers']
            assert result['n_replicas']==4 and result['start']==record['parameters']['start']
            shape=(4,16)
        else:
            cycles=np.asarray(result['connected_spectrum_cycles']);assert cycles.shape==(16,4)
            np.testing.assert_allclose(spectrum,cycles.mean(0),atol=1e-12);shape=(16,)
        ess=np.asarray(result['descendant_ess']);largest=np.asarray(result['largest_lineage'])
        assert ess.shape==largest.shape==shape and np.isfinite(ess).all() and np.isfinite(largest).all()
        assert record['ancestry_alarm']==bool(ess.min()<8 or largest.max()>.5)
        records[key]=record; input_hashes[str(path)]=sha(path)
    rows=[]; contrasts=[]; data={}; diagnostics=[]
    for g in FIELDS:
        for setting in SETTINGS:
            keys=[(g,s,setting,r) for s in IDS for r in range(4)]
            if not all(key in records for key in keys): continue
            group=[records[key] for key in keys]
            raw=np.array([r['result']['connected_spectrum'] for r in group]).reshape(8,4,4)
            exact_raw=np.array([guides[g,s]['exact_spectrum'] for s in IDS]);target=exact_raw.mean(0)
            boot,ix=bootstrap(raw,np.random.default_rng(int(identity(['remedy-assess',g,setting])[:16],16)))
            mean=raw.mean((0,1));check,se=bound_check(mean,boot,target)
            sb=(abs(mean-target)+1.96*se)/target;alarms=sum(r['ancestry_alarm'] for r in group)
            halves=np.array([r['halves'] for r in group]).reshape(8,4,2,4)
            a,b=halves[:,:,0],halves[:,:,1]
            ab=a[np.arange(8)[None,:,None],ix].mean((1,2));bb=b[np.arange(8)[None,:,None],ix].mean((1,2))
            drift,ds=bound_check(b.mean((0,1)),bb,a.mean((0,1)),ab)
            drift_bounds=(abs(b.mean((0,1))-a.mean((0,1)))+1.96*ds)/target
            initial_ok=within(check,sb) and alarms==0
            rows.append(dict(gamma=g,setting=setting,xi=finite(length(mean,2)),exact_xi=finite(length(target,2)),
                mc_se=check['xi_difference_mc_se'],disorder_mc_se=disorder_mc_se(raw),
                disorders=8,repeats=4,spectrum=mean.tolist(),exact_spectrum=target.tolist(),
                ancestry_alarms=alarms,spectral_relative_95_bounds=sb.tolist(),
                exact_check_passes=bool(initial_ok),status='passes_exact_check' if initial_ok else 'unresolved',
                half_drift=drift,half_drift_relative_95_bounds=drift_bounds.tolist(),
                half_drift_passes=within(drift,drift_bounds),ground_state_qualified=False,**check))
            data[g,setting]=(mean,boot,target)
            for sid in IDS:
                rs=[records[g,sid,setting,r] for r in range(4)]
                diagnostics.append(dict(gamma=g,setting=setting,sample_id=sid,
                    minimum_descendant_ess=float(min(np.min(r['result']['descendant_ess']) for r in rs)),
                    maximum_lineage=float(max(np.max(r['result']['largest_lineage']) for r in rs)),
                    ancestry_alarms=sum(r['ancestry_alarm'] for r in rs)))
        for setting in ['population','time','start']:
            if (g,'base') not in data or (g,setting) not in data: continue
            a,ab,target=data[g,'base'];b,bb,other=data[g,setting];np.testing.assert_array_equal(target,other)
            check,se=bound_check(b,bb,a,ab);bounds=(abs(b-a)+1.96*se)/target
            contrasts.append(dict(gamma=g,first='base',second=setting,
                spectral_relative_95_bounds=bounds.tolist(),passes=within(check,bounds),**check))
    decisions=[]
    for g in FIELDS:
        subset=[r for r in rows if r['gamma']==g];cs=[c for c in contrasts if c['gamma']==g]
        passed=len(subset)==4 and len(cs)==3 and all(r['exact_check_passes'] and r['half_drift_passes'] for r in subset) and all(c['passes'] for c in cs)
        decisions.append(dict(gamma=g,passes_development_checks=bool(passed),
                              required_groups=4,complete_groups=len(subset),required_contrasts=3,complete_contrasts=len(cs)))
    ablations=[]
    for g,s in [(.25,50004),(.5,50003),(.75,50005)]:
        for k in ['original_independent','corrected_joint','base']:
            keys=[(g,s,k,r) for r in range(4)]
            if not all(key in records for key in keys): continue
            group=[records[key] for key in keys];raw=np.array([r['result']['connected_spectrum'] for r in group]);target=np.array(guides[g,s]['exact_spectrum'])
            ablations.append(dict(gamma=g,sample_id=s,setting=k,xi=finite(length(raw.mean(0),2)),
                exact_xi=finite(length(target,2)),spectrum=raw.mean(0).tolist(),
                spectral_relative_bias=((raw.mean(0)-target)/target).tolist(),ancestry_alarms=sum(r['ancestry_alarm'] for r in group)))
    report=dict(completed_runs=len(records),expected_runs=len(expected),guide_contexts=len(guides),
        rows=rows,contrasts=contrasts,decisions=decisions,ablations=ablations,disorder_diagnostics=diagnostics,
        manifest_sha256=sha(mp),assessor_sha256=sha(Path(__file__)),input_sha256=input_hashes,
        uncertainty='Fixed-disorder whole-run bootstrap; approximate pointwise 95% bounds. Cycle halves paired within runs; settings independent. Four repeats give limited tail resolution.',
        qualification='L2 development only. No exact labels in guide fitting. Held-out disorders and larger sizes require separate validation; no run is removed for an ancestry alarm.')
    write(root/'validated_review.json',report)
    lines=['# Low-field GFMC remedy review','',report['qualification'],'',report['uncertainty'],'',
        f'Validated {len(records)}/{len(expected)} runs and {len(guides)}/24 guides.','',
        '| Γ | Setting | GFMC ξ | Exact ξ | MC SE | Alarms /32 | Exact check | Half-drift check |',
        '|---:|---|---:|---:|---:|---:|---|---|']
    fmt=lambda x:'unresolved' if x is None else f'{x:.5f}'
    for r in rows:
        lines.append(f"| {r['gamma']} | {r['setting']} | {fmt(r['xi'])} | {fmt(r['exact_xi'])} | {fmt(r['mc_se'])} | {r['ancestry_alarms']} | {r['exact_check_passes']} | {r['half_drift_passes']} |")
    lines+=['','| Γ | Base versus | Δξ | MC SE | Contrast passes |','|---:|---|---:|---:|---|']
    for c in contrasts:
        lines.append(f"| {c['gamma']} | {c['second']} | {fmt(c['xi_difference'])} | {fmt(c['xi_difference_mc_se'])} | {c['passes']} |")
    lines+=['','## Overall development checks','',*['- Γ='+str(d['gamma'])+': '+('passed' if d['passes_development_checks'] else 'unresolved') for d in decisions]]
    (root/'VALIDATED_REVIEW.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();report=assess(args.root)
    print(json.dumps({k:report[k] for k in ['completed_runs','expected_runs','decisions']},indent=2))
