"""Validate and reduce the uniform seven-field corrected-guide disorder ensemble."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from .assess_correlation_escalation import bootstrap, comparison
from .correlation_lengths import ROOT, length, sha
from .correlation_remedy_campaign import write
from .expanded_correlation_campaign import FIELDS, IDS, AUDIT_IDS, SETTINGS, seed_for
from .frozen_scan import SELECTED_SHA256
from .independent_replica_projector import spectrum_from_moments
from .physics import basis, make_af_sample
from .production_protocol import exact_spectra, geometry, identity
from .refine_correlation_guide import features
from .validate_correlation_remedy import label


def finite(value):return float(value) if np.isfinite(value) else None


def passes(check,bounds):
    return bool(check['xi_over_side_95_bound'] is not None and
                check['xi_over_side_95_bound']<=.01 and
                check['invalid_bootstrap_fraction']==0 and np.all(bounds<=.05))


def hierarchical(exact,raw,draws=4000):
    """Whole disorders/repeats, keeping momentum vectors and exact pairing intact."""
    d,r,_=raw.shape;rng=np.random.default_rng(20261102)
    ix=rng.integers(d,size=(draws,d));repeat=rng.integers(r,size=(draws,d,r))
    xb=length(exact[ix].mean(1),2)
    yb=length(raw[ix[...,None],repeat].mean((1,2)),2)
    se=lambda a:finite(np.std(a[np.isfinite(a)],ddof=1))
    return dict(exact_se=se(xb),gfmc_se=se(yb),delta_se=se(yb-xb),
                invalid_bootstrap_fraction=float(np.mean(~np.isfinite(yb))))


def check_group(raw,exact,halves,key,alarms):
    target=exact.mean(0);mean=raw.mean((0,1))
    boot,ix=bootstrap(raw,np.random.default_rng(int(identity(['expanded-assessment',key])[:16],16)))
    check=comparison(mean,boot,target)
    bounds=(abs(mean-target)+1.96*boot.std(0,ddof=1))/target
    a,b=halves[:,:,0],halves[:,:,1];d=raw.shape[0]
    ab=a[np.arange(d)[None,:,None],ix].mean((1,2));bb=b[np.arange(d)[None,:,None],ix].mean((1,2))
    drift=comparison(b.mean((0,1)),bb,a.mean((0,1)),ab)
    db=(abs(b.mean((0,1))-a.mean((0,1)))+1.96*(bb-ab).std(0,ddof=1))/target
    summary=dict(exact=finite(length(target,2)),gfmc=finite(length(mean,2)),
        delta=finite(length(mean,2)-length(target,2)),mc_se=check['xi_difference_mc_se'],
        spectrum=mean.tolist(),exact_spectrum=target.tolist(),spectral_relative_95_bounds=bounds.tolist(),
        ancestry_alarms=int(alarms),exact_check_passes=passes(check,bounds) and alarms==0,
        half_drift_passes=passes(drift,db),half_drift=drift,half_drift_relative_95_bounds=db.tolist(),
        numerical_check=check)
    return summary,(mean,boot,target)


def assess(root,require_complete=False):
    root=Path(root);mp=root/'manifest.json';manifest=json.loads(mp.read_text());signature=identity(manifest)
    assert manifest['gamma']==FIELDS and manifest['sample_ids']==IDS and manifest['audit_ids']==AUDIT_IDS
    assert manifest['settings']==SETTINGS and manifest['repeats']==4
    assert manifest['degree']==4 and manifest['maxiter']==2000 and manifest['exact_labels_used'] is False
    assert manifest['checkpoint_sha256']==SELECTED_SHA256
    hashes={str(mp):sha(mp)}
    for relative,digest in manifest['source_sha256'].items():assert sha(ROOT/relative)==digest,relative
    lattice=geometry(2);spins=basis(12);design,_=features(spins,4);guides={};records={}
    expected_guides={label(g,s) for g in FIELDS for s in IDS}
    with threadpool_limits(limits=1):
        for path in sorted((root/'guides').glob('*.json')):
            gm=json.loads(path.read_text());g,s=gm['gamma'],gm['sample_id'];key=(g,s)
            assert path.stem in expected_guides and path.stem==label(g,s) and key not in guides
            assert gm['signature']==signature and gm['L']==2 and gm['h']==.15
            assert gm['checkpoint_sha256']==SELECTED_SHA256 and gm['exact_labels_used'] is False
            assert sha(path.with_suffix('.npz'))==gm['npz_sha256']
            with np.load(path.with_suffix('.npz')) as a:
                sample=make_af_sample(2,s);h=operator_matrix(sample.hamiltonian(.15),g)
                np.testing.assert_array_equal(a['bonds'],sample.bonds);np.testing.assert_array_equal(a['J'],sample.J)
                logs=a['logs'];coeff=a['coefficients'];psi=a['exact_amplitude']
                assert logs.shape==(4096,) and coeff.shape==(793,) and psi.shape==(4096,)
                assert np.isfinite(logs).all() and np.isfinite(coeff).all() and np.isfinite(psi).all()
                np.testing.assert_allclose(np.exp(2*logs).sum(),1.,atol=1e-10)
                fitted=a['original_logs']+design@coeff
                np.testing.assert_allclose(fitted-fitted.max(),logs-logs.max(),atol=1e-10)
                np.testing.assert_allclose(psi@psi,1.,atol=1e-10)
                assert gm['residual']<1e-9 and np.linalg.norm(h@psi-gm['exact_energy']*psi)<1e-9
                spectrum=exact_spectra(psi**2,spins,lattice)['connected_spectrum']
                np.testing.assert_allclose(spectrum,gm['exact_spectrum'],atol=1e-10)
            guides[key]=gm;hashes[str(path)]=sha(path);hashes[str(path.with_suffix('.npz'))]=sha(path.with_suffix('.npz'))
    expected={(g,s,k,r) for k in SETTINGS for g in FIELDS for s in (IDS if k=='production' else AUDIT_IDS) for r in range(4)}
    for path in sorted((root/'runs').glob('*.json')):
        record=json.loads(path.read_text());g,s,k,r=[record[n] for n in ['gamma','sample_id','setting','repeat']];key=(g,s,k,r)
        assert key in expected and key not in records and path.name==label(g,s)+f'_{k}_r{r}.json'
        assert record['signature']==signature and record['L']==2 and record['h']==.15
        assert record['seed']==seed_for(g,s,k,r) and record['parameters']==SETTINGS[k]
        assert record['guide_sha256']==guides[g,s]['npz_sha256']
        assert record['exact_spectrum']==guides[g,s]['exact_spectrum'] and record['ground_state_qualified'] is False
        result=record['result'];m=np.asarray(result['magnetization_cycles']);q=np.asarray(result['pair_cycles'])
        assert m.shape==(4,16,12) and q.shape==(4,16,12,12) and np.isfinite(m).all() and np.isfinite(q).all()
        assert np.max(np.abs(m))<=1+1e-12 and np.max(np.abs(q))<=1+1e-12
        np.testing.assert_allclose(np.diagonal(q,axis1=-2,axis2=-1),1.,atol=1e-12)
        spectrum=spectrum_from_moments(m.mean(1),q.mean(1),lattice)
        np.testing.assert_allclose(spectrum,result['connected_spectrum'],atol=1e-12)
        np.testing.assert_allclose(result['chi_connected'],spectrum[0],atol=1e-12)
        for n in ['seed','n_walkers','start']:
            assert result[n]==(record['seed'] if n=='seed' else SETTINGS[k][n])
        assert result['n_replicas']==4
        ess=np.asarray(result['descendant_ess']);largest=np.asarray(result['largest_lineage']);unique=np.asarray(result['unique_ancestors'])
        assert ess.shape==largest.shape==unique.shape==(4,16)
        assert np.isfinite(ess).all() and np.isfinite(largest).all()
        assert np.all((ess>=1)&(ess<=SETTINGS[k]['n_walkers']+1e-10))
        assert np.all((largest>0)&(largest<=1)) and np.all((unique>=1)&(unique<=SETTINGS[k]['n_walkers']))
        assert result['ancestry_alarm']==bool(ess.min()<8 or largest.max()>.5)
        record['halves']=[spectrum_from_moments(m[:,sl].mean(1),q[:,sl].mean(1),lattice) for sl in [slice(0,8),slice(8,16)]]
        records[key]=record;hashes[str(path)]=sha(path)
    if require_complete:assert len(guides)==448 and set(records)==expected,'Incomplete expanded campaign'
    rows=[];audit_rows=[];contrasts=[];diagnostics=[];raw_csv=[]
    for g in FIELDS:
        rectangle=lambda ids,k:all((g,s,k,r) in records for s in ids for r in range(4))
        def group(ids,k):
            rs=[records[g,s,k,r] for s in ids for r in range(4)]
            raw=np.array([r['result']['connected_spectrum'] for r in rs]).reshape(len(ids),4,4)
            exact=np.array([guides[g,s]['exact_spectrum'] for s in ids])
            halves=np.array([r['halves'] for r in rs]).reshape(len(ids),4,2,4)
            alarms=sum(r['result']['ancestry_alarm'] for r in rs)
            return rs,raw,exact,halves,alarms
        if rectangle(IDS,'production'):
            rs,raw,exact,halves,alarms=group(IDS,'production')
            row,_=check_group(raw,exact,halves,(g,'production'),alarms)
            row.update(hierarchical(exact,raw));row.update(gamma=g,L=2,h=.15,disorders=len(IDS),repeats=4)
            row['minimum_descendant_ess']=float(min(np.min(r['result']['descendant_ess']) for r in rs))
            prefix=[]
            for n in [8,16,32,64]:
                prefix.append(dict(disorders=n,exact=finite(length(exact[:n].mean(0),2)),gfmc=finite(length(raw[:n].mean((0,1)),2)),**hierarchical(exact[:n],raw[:n])))
            row['disorder_prefixes']=prefix;rows.append(row)
            for i,s in enumerate(IDS):
                subset=rs[i*4:(i+1)*4];mean=raw[i].mean(0)
                diagnostics.append(dict(gamma=g,sample_id=s,ancestry_alarms=sum(r['result']['ancestry_alarm'] for r in subset),
                    spectral_relative_bias=((mean-exact[i])/exact[i]).tolist(),
                    minimum_descendant_ess=float(min(np.min(r['result']['descendant_ess']) for r in subset))))
        audits={}
        for k in SETTINGS:
            if not rectangle(AUDIT_IDS,k):continue
            rs,raw,exact,halves,alarms=group(AUDIT_IDS,k)
            row,data=check_group(raw,exact,halves,(g,'audit',k),alarms);row.update(gamma=g,setting=k,disorders=8,repeats=4)
            audit_rows.append(row);audits[k]=data
        if 'production' in audits:
            a,ab,target=audits['production']
            for k in ['population','time']:
                if k not in audits:continue
                b,bb,other=audits[k];np.testing.assert_array_equal(target,other)
                check=comparison(b,bb,a,ab);bounds=(abs(b-a)+1.96*(bb-ab).std(0,ddof=1))/target
                contrasts.append(dict(gamma=g,second=k,passes=passes(check,bounds),spectral_relative_95_bounds=bounds.tolist(),**check))
        for row in [row for row in rows if row['gamma']==g]:
            ar=[r for r in audit_rows if r['gamma']==g];cs=[c for c in contrasts if c['gamma']==g]
            passed=(row['exact_check_passes'] and row['half_drift_passes'] and len(ar)==3 and len(cs)==2
                    and all(r['exact_check_passes'] and r['half_drift_passes'] for r in ar) and all(c['passes'] for c in cs))
            row['comparison_status']='passes L2 ensemble and audit checks' if passed else 'unresolved'
            row['checks_pass']=bool(passed)
    for key,r in sorted(records.items()):
        raw_csv.append([*key,*r['result']['connected_spectrum'],r['result']['ancestry_alarm']])
    report=dict(completed_runs=len(records),expected_runs=len(expected),completed_guides=len(guides),expected_guides=448,
        complete_fields=len(rows),rows=rows,audit_rows=audit_rows,contrasts=contrasts,disorder_diagnostics=diagnostics,
        ancestry_alarms=sum(r['result']['ancestry_alarm'] for r in records.values()),
        all_checks_pass=len(rows)==7 and all(r['checks_pass'] for r in rows),input_sha256=hashes,
        manifest_sha256=sha(mp),assessor_sha256=sha(Path(__file__)),
        scope=manifest['scope'],uncertainty='4000 hierarchical bootstrap draws for plotted errors; fixed-disorder whole-repeat bootstrap for approximate pointwise 95% numerical checks. Four repeats limit tail resolution.',
        qualification='Reference agreement on all 64 disorders per field; population/time audits on eight prespecified disorders per field. No large-size qualification.')
    write(root/'validated_review.json',report)
    with (root/'raw_spectra.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['gamma','sample_id','setting','repeat','S0','Sk1','Sk2','Sk3','ancestry_alarm']);writer.writerows(raw_csv)
    columns=['gamma','L','h','disorders','repeats','exact','exact_se','gfmc','gfmc_se','delta','delta_se',
             'mc_se','ancestry_alarms','invalid_bootstrap_fraction','comparison_status','checks_pass']
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Uniform seven-field corrected-guide benchmark','',report['scope'],'',report['uncertainty'],'',report['qualification'],'',
           f'Validated {len(records)}/{len(expected)} runs; {len(rows)}/7 complete 64-disorder fields.','',
           '| Γ | Exact ξ | GFMC ξ | Disorder + MC SE | Alarms /256 | Check |','|---:|---:|---:|---:|---:|---|']
    fmt=lambda x:'unresolved' if x is None else f'{x:.6f}'
    for r in rows:lines.append(f"| {r['gamma']} | {fmt(r['exact'])} | {fmt(r['gfmc'])} | {fmt(r['gfmc_se'])} | {r['ancestry_alarms']} | {r['comparison_status']} |")
    lines.extend(['','## Disorder-prefix stability','',
        'Each prefix uses the first n matched disorders, with spectra averaged before computing the length. Prefixes are nested; their errors are correlated.', '',
        '| Γ | Disorders | Exact ξ ± SE | GFMC ξ ± SE |',
        '|---:|---:|---:|---:|'])
    for r in rows:
        for prefix in r['disorder_prefixes']:
            lines.append(f"| {r['gamma']} | {prefix['disorders']} | {fmt(prefix['exact'])} ± {fmt(prefix['exact_se'])} | {fmt(prefix['gfmc'])} ± {fmt(prefix['gfmc_se'])} |")
    lines.extend(['','## Population and time contrasts','',
        'Approximate pointwise 95% upper bounds on the absolute change, conditional on the eight prespecified audit disorders. Spectrum bounds are relative to the matched exact spectra. Passing these contrasts also requires the separate reference, drift and ancestry checks recorded in validated_review.json.', '',
        '| Γ | Audit | Bound on Δξ/(2L) | Largest relative spectral bound | Pass |',
        '|---:|---|---:|---:|---|'])
    for contrast in contrasts:
        lines.append(f"| {contrast['gamma']} | {contrast['second']} | {fmt(contrast['xi_over_side_95_bound'])} | {fmt(max(contrast['spectral_relative_95_bounds']))} | {contrast['passes']} |")
    (root/'VALIDATED_REVIEW.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    report=assess(a.root,a.require_complete)
    print(json.dumps({k:report[k] for k in ['completed_runs','expected_runs','complete_fields','ancestry_alarms','all_checks_pass']},indent=2))
