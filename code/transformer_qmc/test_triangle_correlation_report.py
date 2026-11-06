"""Coverage and matched-disorder checks for the live scaling report."""
import json

import numpy as np

from .triangle_correlation_report import report
from .triangle_correlation_grid import GAMMAS


def test_complete_groups_and_retention(tmp_path):
    # Four fields at two sizes; partial L8 coverage must not enter the report.
    for L in (4, 6, 8):
        for field in (.1, .2):
            for chi in (2, 3):
                for sample_id in range(60000, 60008 if L != 8 else 60002):
                    folder=tmp_path/'cases'/f'L{L}_g{field:g}_s{sample_id}_m6_c{chi}'
                    folder.mkdir(parents=True)
                    scale=.05*(L-4)+.02*(field-.1)+.001*(chi-2)+.0005*(sample_id-60000)
                    spectrum=np.array([1.1+scale,1.,1.,1.])
                    n=3*L*L
                    zz=np.eye(n)
                    np.savez_compressed(folder/'seed0_D64.npz',zz=zz)
                    case=dict(L=L,gamma=field,sample_id=sample_id,chi=chi,solver_passed=True,
                        records=[dict(seed=0,D_MPS=64,energy=-10.,spectra=spectrum.tolist(),R=.1),
                                 dict(seed=1,D_MPS=64,energy=-9.,spectra=spectrum.tolist(),R=.1)])
                    (folder/'result.json').write_text(json.dumps(case))
    report(tmp_path)
    summary=json.loads((tmp_path/'report/summary.json').read_text())
    assert len(summary['complete_disorder_groups']) == 8
    assert all(row['L'] != 8 for row in summary['complete_disorder_groups'])
    assert all(row['q2'] == 1/(3*row['L']*row['L']) for row in summary['complete_disorder_groups'])
    assert len(summary['feasibility_two_disorders']) == 12
    assert summary['candidate_crossings'] == []
    assert len(summary['retention']) == 4


def test_dense_crossings_do_not_bridge_missing_fields(tmp_path):
    assert len(GAMMAS) == 57
    assert all(round(i/10, 1) in GAMMAS for i in range(1, 16))
    # Two adjacent sign changes; the final sign change spans missing Gamma=.175.
    for field, difference in ((.1, .04), (.125, -.04), (.15, .04), (.2, -.04)):
        for L in (4, 6):
            spectrum=[1.2+(difference if L == 6 else 0), 1., 1., 1.]
            for sample_id in range(60000, 60008):
                folder=tmp_path/'cases'/f'L{L}_g{field:g}_s{sample_id}_m6_c2'
                folder.mkdir(parents=True)
                np.savez_compressed(folder/'seed0_D64.npz', zz=np.eye(3*L*L))
                case=dict(L=L,gamma=field,sample_id=sample_id,chi=2,solver_passed=True,
                    records=[dict(seed=seed,D_MPS=64,energy=-10.+seed,
                                  spectra=spectrum,R=.1) for seed in (0,1)])
                (folder/'result.json').write_text(json.dumps(case))
    report(tmp_path)
    summary=json.loads((tmp_path/'report/summary.json').read_text())
    brackets=[(r['lower'],r['upper']) for r in summary['candidate_crossings']]
    assert brackets == [(.1,.125),(.125,.15)]
    assert summary['gammas'] == GAMMAS
