"""Local, reproducible diagnostics of the completed L6 chi=8 D32/D64 scan.

Reads downloaded JSON and physical moment arrays; never runs a solver.
Qualification thresholds reproduce the archived solver source, including
its asymmetric spectrum denominators and strict versus inclusive bounds.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_NAME = 'triangle_L6_m6_chi8_D64_diagnostics_20261106'
OBSERVABLES = ('q_ea_finite', 'q2_raw', 'chi_sg_connected_equal_time',
               'chi_sg_connected_diagonal', 'chi_sg_connected_off_diagonal',
               'chi_sg_raw_equal_time', 'magnetization')
GATE_LABELS = {
    'energy': 'Energy per spin', 'spectrum': 'Connected structure factors',
    'q': 'q_EA', 'q2': '<q²>', 'connected': 'Connected susceptibility',
    'R': 'Correlation-length ratio R', 'last_energy': 'Last sweep energy',
    'canonical': 'MPS canonical error', 'truncation': 'Current truncation alarm',
    'prior_truncation': 'Prior truncation alarm',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moment_values(a):
    m, zz = a['magnetization'], a['zz']
    assert m.shape == (108,) and zz.shape == (108, 108)
    assert np.isfinite(m).all() and np.isfinite(zz).all()
    np.testing.assert_allclose(zz, zz.T, atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.diag(zz), 1, atol=1e-8, rtol=0)
    connected = zz - np.outer(m, m)
    diag = np.mean(np.diag(connected)**2)
    total = np.sum(connected**2)/108
    return dict(q_ea_finite=float(np.mean(m*m)), q2_raw=float(np.mean(zz*zz)),
                magnetization=float(np.mean(m)), chi_sg_connected_equal_time=float(total),
                chi_sg_connected_diagonal=float(diag),
                chi_sg_connected_off_diagonal=float(total-diag),
                chi_sg_raw_equal_time=float(np.sum(zz*zz)/108))


def compare(a, b, kind, gamma, sid, seed):
    """a=new D64, b=D32 for ladder; a=seed0, b=seed1 for starts."""
    ladder = kind == 'ladder'
    av, bv = a['observables'], b['observables']
    def rel(key):
        return abs(av[key]-bv[key])/max(abs(av[key]), abs(bv[key]), 1e-10)
    sa, sb = np.asarray(a['spectra']), np.asarray(b['spectra'])
    metrics = dict(energy=abs(a['energy']-b['energy'])/108,
                   spectrum=float(np.max(abs(sa-sb)/np.maximum(abs(sa), 1e-10))),
                   q=abs(av['q_ea_finite']-bv['q_ea_finite']),
                   q2=rel('q2_raw'), connected=rel('chi_sg_connected_equal_time'),
                   R=abs(a['R']-b['R']) if a['R'] is not None and b['R'] is not None else np.nan)
    limits = dict(energy=1e-7 if ladder else 1e-6,
                  spectrum=.005 if ladder else .01, q=.001,
                  q2=.005 if ladder else .01, connected=.005 if ladder else .01, R=.002)
    out = dict(gamma=gamma, sample_id=sid, kind=kind, seed=seed,
               q_signed=av['q_ea_finite']-bv['q_ea_finite'],
               q2_signed=av['q2_raw']-bv['q2_raw'])
    gates = {}
    for key, value in metrics.items():
        out[key+'_delta'] = value
        out[key+'_limit'] = limits[key]
        out[key+'_ratio'] = value/limits[key]
        gates[key] = bool(np.isfinite(value) and (value <= limits[key] if key in ('q','q2','connected') else value < limits[key]))
    if ladder:
        gates.update(last_energy=a['last_energy_change']/108 < 1e-8,
                     canonical=a['canonical_error'] < 1e-8,
                     truncation=a['truncation_failure'] is None,
                     prior_truncation=b.get('truncation_failure') is None)
        out.update(last_energy_delta=a['last_energy_change']/108,
                   last_energy_limit=1e-8, last_energy_ratio=a['last_energy_change']/108/1e-8,
                   canonical_delta=a['canonical_error'], canonical_limit=1e-8,
                   canonical_ratio=a['canonical_error']/1e-8)
    out.update({key+'_pass': bool(value) for key, value in gates.items()})
    out['all_pass'] = all(gates.values())
    out['failed_gates'] = ', '.join(key for key, value in gates.items() if not value)
    return out


def analyze(root):
    root = Path(root)
    data = root/'data/kagome'/DATA_NAME
    raw = data/'raw'
    source = raw/'source/code/transformer_qmc'
    sources = {p.name:digest(p) for p in source.glob('*.py')}
    expected = {(i/1000,s) for i in range(100,1501,25) for s in range(60000,60008)}
    paths = sorted((raw/'results/cases').glob('*/result.json'))
    assert len(paths) == 456
    records, comparisons, cases, inventory = [], [], [], []
    disorder_hashes = {}
    seen = set()
    for p in paths:
        d = json.loads(p.read_text())
        g, sid = d['gamma'], d['sample_id']
        assert (g,sid) not in seen
        seen.add((g,sid))
        assert (d['L'],d['m'],d['chi'],d['h'],d['lambda_rg']) == (6,6,8,.15,1.)
        assert d['D_ladder'] == [32,64] and d['max_bond_dimension'] == 64
        assert d['computation_complete'] and d['overlap_gates_enabled']
        for name, sha in d['solver_source_sha256'].items():
            assert sources[name] == sha, (name,sha)
        inventory.append(dict(path=str(p.relative_to(data)),sha256=digest(p)))
        lookup = {}
        for rec in d['records']:
            seed, dim = rec['seed'], rec['D_MPS']
            assert (seed,dim) not in lookup
            name = f'seed{seed}_D{dim}.npz'
            moment_path = p.parent/name
            assert digest(moment_path) == d['hashes'][name], moment_path
            if 'checkpoint_hashes' in rec:
                assert rec['checkpoint_hashes'][name] == d['hashes'][name]
            stage = json.loads((p.parent/f'seed{seed}_D{dim}.json').read_text())
            assert stage == rec
            with np.load(moment_path, allow_pickle=False) as a:
                obs = moment_values(a)
                for key in OBSERVABLES:
                    np.testing.assert_allclose(obs[key],rec['observables'][key],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(obs['chi_sg_connected_equal_time'],a['spectra'][0],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(rec['spectra'],a['spectra'],atol=1e-10,rtol=1e-10)
                np.testing.assert_array_equal(a['disorder_entropy'],[20261027,6,sid,1])
                assert np.all((-a['J'] >= .5) & (-a['J'] <= 1.5))
                bond_hash = hashlib.sha256(a['bonds'].tobytes()+a['J'].tobytes()).hexdigest()
                if sid in disorder_hashes:
                    assert disorder_hashes[sid] == bond_hash
                disorder_hashes[sid] = bond_hash
            # Use the archived JSON numbers for exact reproduction of threshold decisions;
            # independently recomputed moment values agree above.
            lookup[seed,dim] = rec
            inventory.append(dict(path=str(moment_path.relative_to(data)),sha256=d['hashes'][name]))
            records.append(dict(gamma=g,sample_id=sid,seed=seed,D_MPS=dim,
                energy=rec['energy'],energy_per_spin=rec['energy']/108,
                R=rec['R'],max_bond=rec['max_bond'],sweeps=rec['sweeps'],
                last_energy_change_per_spin=rec['last_energy_change']/108,
                canonical_error=rec['canonical_error'],truncation_failure=rec['truncation_failure'],
                ladder_stable=rec['ladder_stable'],solver_passed=d['solver_passed'],
                **rec['observables']))
        assert set(lookup) == {(s,D) for s in (0,1) for D in (32,64)}
        local = []
        for seed in (0,1):
            c = compare(lookup[seed,64],lookup[seed,32],'ladder',g,sid,seed)
            assert c['all_pass'] == lookup[seed,64]['ladder_stable'], (g,sid,seed,c)
            local.append(c)
        c = compare(lookup[0,64],lookup[1,64],'starts',g,sid,-1)
        assert c['all_pass'] == d['start_agreement'], (g,sid,c)
        local.append(c)
        passed = all(c['all_pass'] for c in local)
        assert passed == d['solver_passed'], (g,sid)
        audit_path = raw/'results/audits'/f'g{round(g*1000):04d}_s{sid}.json'
        audit = json.loads(audit_path.read_text())
        assert audit['result_sha256'] == digest(p)
        assert audit['qualified'] == passed
        assert audit['bonds_sha256'] == bond_hash
        comparisons.extend(local)
        row = dict(gamma=g,sample_id=sid,solver_passed=passed,
                   ladder_pass=all(c['all_pass'] for c in local[:2]),starts_pass=local[-1]['all_pass'])
        for key in GATE_LABELS:
            applicable = [c for c in local if key+'_pass' in c]
            row[key+'_pass'] = all(c[key+'_pass'] for c in applicable)
            row[key+'_ladder_pass'] = all(c[key+'_pass'] for c in local[:2])
            row[key+'_starts_pass'] = local[-1].get(key+'_pass',True)
        row['overlap_only_pass'] = all(row[key+'_pass'] for key in ('q','q2','connected'))
        row['q_and_q2_pass'] = row['q_pass'] and row['q2_pass']
        cases.append(row)
    assert seen == expected
    records, comparisons, cases = [pd.DataFrame(x).sort_values(['gamma','sample_id']).reset_index(drop=True)
                                   for x in (records,comparisons,cases)]
    summary = dict(cases=len(cases),records=len(records),comparisons=len(comparisons),
                   solver_passed=int(cases.solver_passed.sum()),solver_failed=int((~cases.solver_passed).sum()),
                   ladder_failed=int((~cases.ladder_pass).sum()),starts_failed=int((~cases.starts_pass).sum()),
                   gate_failure_cases={k:int((~cases[k+'_pass']).sum()) for k in GATE_LABELS},
                   q_only_pass=int(cases.q_pass.sum()),q_and_q2_pass=int(cases.q_and_q2_pass.sum()),
                   overlap_only_pass=int(cases.overlap_only_pass.sum()),
                   full_fail_but_q_pass=int((~cases.solver_passed & cases.q_pass).sum()),
                   full_fail_but_overlap_pass=int((~cases.solver_passed & cases.overlap_only_pass).sum()),
                   source_sha256=sources,disorder_sha256=disorder_hashes,
                   archive_sha256=digest(data/'evidence.tar.gz'),
                   verified_utc=datetime.now(timezone.utc).isoformat(),
                   remote_root='/root/autodl-tmp/triangle-L6-m6-c8-D64-dense-20261105-v1',
                   interpretation='Finite D32-to-D64 sensitivity and start agreement; not proof of asymptotic nonconvergence or physical retention convergence.')
    assert summary['solver_passed'] == 111 and summary['solver_failed'] == 345
    for name, frame in [('records',records),('comparisons',comparisons),('cases',cases)]:
        frame.to_csv(data/f'{name}.csv',index=False)
    (data/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (data/'verified_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    return records, comparisons, cases, summary


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[2]
    _, comp, cases, summary = analyze(repo)
    print(json.dumps(summary,indent=2))
    for kind in ('ladder','starts'):
        c = comp[comp.kind == kind]
        print(kind, c[['q_delta','q2_delta','connected_delta','energy_delta','R_delta']].describe().to_string())
        print(c.nlargest(5,'q_delta')[['gamma','sample_id','seed','q_delta','q2_delta','failed_gates']].to_string(index=False))
