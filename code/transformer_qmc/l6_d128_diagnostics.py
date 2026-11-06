"""Verify partial or complete D128 continuations against the local D64 archive."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .l6_d64_diagnostics import DATA_NAME, compare, digest, moment_values

DATA128 = 'triangle_L6_m6_chi8_D128_diagnostics_20261106'


def analyze(root):
    root=Path(root)
    oldroot=root/'data/kagome'/DATA_NAME/'raw/results/cases'
    data=root/'data/kagome'/DATA128
    rows=[]; comparisons=[]; cases=[]
    for p in sorted((data/'raw/results/cases').glob('*/result.json')):
        d=json.loads(p.read_text())
        oldpath=oldroot/p.parent.name/'result.json'
        old=json.loads(oldpath.read_text())
        assert digest(oldpath)==d['ancestry']['baseline_sha256']
        assert d['max_bond_dimension']==128 and d['D_ladder']==[32,64,128]
        assert d['computation_complete'] and not d['retention_qualified']
        assert (d['L'],d['m'],d['chi'],d['h'],d['lambda_rg'])==(6,6,8,.15,1.)
        for filename,sha in d['ancestry']['source_sha256'].items():
            assert digest(data/'raw/source/code/transformer_qmc'/filename)==sha
        lookup={(r['seed'],r['D_MPS']):r for r in d['records']}
        assert set(lookup)=={(s,D) for s in (0,1) for D in (32,64,128)}
        for rec in old['records']:
            assert lookup[rec['seed'],rec['D_MPS']]==rec
        for seed in (0,1):
            rec=lookup[seed,128];stem=f'seed{seed}_D128'
            assert rec==json.loads((p.parent/(stem+'.json')).read_text())
            assert rec['parent_checkpoint_sha256']==old['hashes'][f'seed{seed}_D64.pkl']
            moment=p.parent/(stem+'.npz')
            assert digest(moment)==d['hashes'][moment.name]==rec['checkpoint_hashes'][moment.name]
            with np.load(moment) as a, np.load(oldpath.parent/f'seed{seed}_D64.npz') as b:
                for key in ['bonds','J','points','disorder_entropy']:
                    np.testing.assert_array_equal(a[key],b[key])
                obs=moment_values(a)
                for key,value in obs.items():
                    np.testing.assert_allclose(value,rec['observables'][key],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(a['spectra'],rec['spectra'],atol=1e-10,rtol=1e-10)
        local=[compare(lookup[s,128],lookup[s,64],'ladder',d['gamma'],d['sample_id'],s) for s in (0,1)]
        local.append(compare(lookup[0,128],lookup[1,128],'starts',d['gamma'],d['sample_id'],-1))
        for i,comp in enumerate(local):
            assert comp['all_pass']==d['comparisons'][i]['all_pass']
            if i<2:assert comp['all_pass']==lookup[i,128]['ladder_stable']
        assert all(c['all_pass'] for c in local)==d['solver_passed']
        comparisons.extend(local)
        case=dict(gamma=d['gamma'],sample_id=d['sample_id'],solver_passed=d['solver_passed'],
                  prior_solver_passed=old['solver_passed'])
        for key in ['q','q2','connected','energy','spectrum','R']:
            case[key+'_pass']=all(c[key+'_pass'] for c in local)
        cases.append(case)
        for rec in d['records']:
            rows.append(dict(gamma=d['gamma'],sample_id=d['sample_id'],seed=rec['seed'],D_MPS=rec['D_MPS'],
                             energy_per_spin=rec['energy']/108,R=rec['R'],
                             truncation_failure=rec['truncation_failure'],**rec['observables']))
    records=pd.DataFrame(rows); comp=pd.DataFrame(comparisons); cases=pd.DataFrame(cases)
    summary=dict(target=456,completed=len(cases),qualified=int(cases.solver_passed.sum()) if len(cases) else 0,
                 q_pass=int(cases.q_pass.sum()) if len(cases) else 0,
                 full_fields=[float(g) for g,n in cases.groupby('gamma').size().items() if n==8] if len(cases) else [],
                 snapshot=json.loads((data/'raw/results/status.json').read_text()))
    for label,frame in [('records',records),('comparisons',comp),('cases',cases)]:
        frame.to_csv(data/(label+'.csv'),index=False)
    (data/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return records,comp,cases,summary
