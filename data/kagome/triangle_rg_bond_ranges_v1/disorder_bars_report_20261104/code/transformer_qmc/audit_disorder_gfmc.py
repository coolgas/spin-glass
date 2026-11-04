"""Verify coverage, provenance and retained diagnostics of a disorder campaign."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .experiment import write_json
from .frozen_scan import digest
from .physics import make_af_sample


def audit(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    snapshot=root/'source_snapshot'
    source_checked=False
    if snapshot.exists():
        for name,checksum in manifest['source_sha256'].items():
            assert digest(snapshot/name)==checksum,name
        source_checked=True
    cache={}
    for p in (root/'cache').glob('*.json'):
        r=json.loads(p.read_text())
        assert digest(p.with_suffix('.npz'))==r['cache_sha256'],p
        assert r['residual']<1e-9,p
        with np.load(p.with_suffix('.npz')) as data:
            sample=make_af_sample(2,r['sample_id'])
            np.testing.assert_array_equal(data['bonds'],sample.bonds)
            np.testing.assert_array_equal(data['J'],sample.J)
            np.testing.assert_allclose(np.exp(2*data['logs']).sum(),1,atol=1e-10)
            np.testing.assert_allclose(data['exact']@data['exact'],1,atol=1e-10)
        cache[(r['sample_id'],r['h'],r['gamma'])]=r
    expected_cache=set(itertools.product(manifest['sample_ids'],manifest['h'],manifest['gamma']))
    assert set(cache)<=expected_cache
    expected=set()
    for setting in manifest['settings']:
        ids=manifest['sample_ids'] if setting=='main' else manifest['audit_ids']
        expected.update(itertools.product(ids,manifest['h'],manifest['gamma'],[setting],
                                          ['constant','network'],manifest['seeds']))
    seen=set()
    for p in (root/'populations').glob('*.json'):
        r=json.loads(p.read_text())
        identity=tuple(r[k] for k in ('sample_id','h','gamma','setting','guide','seed'))
        assert identity in expected and identity not in seen,p
        seen.add(identity)
        assert r['checkpoint_sha256']==manifest['checkpoint_sha256'],p
        assert r['cache_sha256']==cache[identity[:3]]['cache_sha256'],p
        assert r['parameters']==manifest['settings'][r['setting']],p
        cycles=r['parameters']['cycles']; w=r['parameters']['walkers']
        for name in ('q','q2','q4','chi_connected'):
            values=np.array(r[name+'_cycles'])
            assert len(values)==cycles and np.isfinite(values).all(),p
            np.testing.assert_allclose(values.mean(),r[name],atol=1e-12)
        for name in ('descendant_ess','unique_ancestors','largest_lineage'):
            assert len(r[name])==cycles and np.isfinite(r[name]).all(),p
        assert min(r['descendant_ess'])>=1-1e-9 and max(r['descendant_ess'])<=w+1e-9,p
        for name in ('energy','energy_growth','seconds'):
            assert np.isfinite(r[name]),p
    references=set()
    for p in (root/'finite_forward').glob('*.json'):
        r=json.loads(p.read_text())
        identity=(r['sample_id'],r['h'],r['gamma'])
        assert identity not in references,p
        assert r['cache_sha256']==cache[identity]['cache_sha256'],p
        assert len(r['rows'])==6,p
        assert {(row['setting'],row['guide']) for row in r['rows']}==set(itertools.product(
            ['main','equilibration','forward'],['constant','network'])),p
        for row in r['rows']:
            values=np.asarray(row['cycles'])
            assert values.shape==(16,4) and np.isfinite(values).all(),p
            np.testing.assert_allclose(values.mean(0),[row['mean'][k] for k in ('q','q2','chi_sg','chi_connected')])
        references.add(identity)
    report=dict(caches=len(cache),expected_caches=448,populations=len(seen),
                expected_populations=len(expected),finite_forward=len(references),
                source_snapshot_verified=source_checked,
                complete=set(cache)==expected_cache and seen==expected and references==expected_cache,
                passed=True,checkpoint_sha256=manifest['checkpoint_sha256'])
    write_json(root/'audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.results),indent=2))


if __name__=='__main__':
    main()
