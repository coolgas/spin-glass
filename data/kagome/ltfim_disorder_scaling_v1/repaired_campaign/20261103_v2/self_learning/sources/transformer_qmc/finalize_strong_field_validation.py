"""Verify and package completed repair controls on AutoDL."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();run=args.run;dest=args.destination
    validation=run/'validation'
    fits=[]
    for directory,stem in [
        ('training','L2_s60000_g4_rbm_2_direct_seed0'),
        ('training','L6_s60000_g4_rbm_2_direct_seed0'),
        ('training','L6_s60000_g1_rbm_2_anneal_seed0'),
        ('jastrow_stable/training','L6_s60000_g4_jastrow_direct_seed0')]:
        p=validation/directory/(stem+'.json')
        r=json.loads(p.read_text());assert r['status']=='completed'
        assert digest(p.with_suffix('.pt'))==r['checkpoint_sha256']
        assert digest(p.with_suffix('.npz'))==r['npz_sha256']
        fits.append(dict(path=str(p.relative_to(validation)),task=r['task'],
                         learning_rate=r['signature']['learning_rate'],
                         sampling_alarms=r['sampling_alarms']))
    projections=[];verified=[]
    for L,kind in [(2,'rbm_2'),(6,'rbm_2'),(6,'jastrow')]:
        for audit in ['main','start','population','time']:
            for repeat in [0,1]:
                p=validation/'projector'/f'L{L}_{kind}_{audit}_r{repeat}.json'
                r=json.loads(p.read_text())
                assert digest(p.with_suffix('.npz'))==r['raw_sha256']
                assert digest(Path(r['case']['checkpoint']))==r['checkpoint_sha256']
                with np.load(p.with_suffix('.npz')) as raw:
                    tags=raw['tags'].astype(float);counts=raw['counts'];w=r['case']['walkers']
                    assert np.issubdtype(counts.dtype,np.integer) and (counts>=0).all()
                    np.testing.assert_equal(counts.sum(axis=2),w)
                    local=np.einsum('crw,crwn->crn',counts/w,tags).mean(axis=0)
                    q=np.mean([np.mean(local[a]*local[b]) for a,b in combinations(range(4),2)])
                    np.testing.assert_allclose(q,r['mean_q'],rtol=0,atol=1e-14)
                    assert np.isfinite(raw['energy']).all() and np.isfinite(raw['growth']).all()
                    np.testing.assert_allclose(raw['descendant_ess'].min(),r['minimum_descendant_ess'],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(raw['largest_lineage'].max(),r['largest_lineage'],rtol=0,atol=1e-12)
                verified.append(p.name)
                projections.append({k:v for k,v in r.items() if k!='numerical_trace'})
    assert len(verified)==24 and len(set(verified))==24
    (run/'raw_validation_checks.json').write_text(json.dumps(dict(verified=verified,
        checks=['hash','q cross-population identity','counts normalization','finite energies',
                'descendant ESS','largest lineage']),indent=2)+'\n')
    (validation/'projection_summary.json').write_text(json.dumps(dict(results=projections),indent=2)+'\n')
    (validation/'projection_complete.json').write_text(json.dumps(dict(completed=24,failures=[],ground_state_qualified=False),indent=2)+'\n')
    (validation/'training_recovered_summary.json').write_text(json.dumps(dict(results=fits,
        initial_failures='training_summary_initial_failure.json; original interrupted Jastrow status collision retained',
        ground_state_qualified=False),indent=2)+'\n')
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['optimizer','optimizer_sources','validation','validation_sources','executed_sources','preserved_failure']:
        shutil.copytree(run/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    sources=dest/'repair_sources';sources.mkdir(exist_ok=True)
    module=Path(__file__).parent
    for name in ['strong_field_validation.py','strong_field_report.py','finalize_strong_field_validation.py',
                 'reproduce_rbm_underflow.py','sampled_simple_guides.py','projector.py','disorder_scaling.py',
                 'parallel_disorder_scaling.py','simple_guides.py','physics.py','test_log_projector.py',
                 'test_sampled_simple_guides.py','test_training_priority.py']:
        shutil.copy2(module/name,sources/name)
    shutil.copy2(module.parent/'qmc_numerics.py',sources/'qmc_numerics.py')
    for p in (list(run.glob('*.log'))+list(run.glob('*launch.json'))+list(run.glob('regression*.txt'))+
              [run/'failed_start_reproduction.json',run/'raw_validation_checks.json',run/'README.md']):
        shutil.copy2(p,dest/p.name)
    # Source hashes resolve to an actual preserved file, including older fit snapshots.
    lookup={digest(p) for p in dest.rglob('*.py')}
    for item in fits:
        r=json.loads((validation/item['path']).read_text())
        assert set(r['signature']['sources'].values()).issubset(lookup),item['path']
    for r in projections:
        assert set(r['projection_sources'].values()).issubset(lookup),r['case']['name']
        assert r['numerics_sha256'] in lookup
    hashes={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*'))
            if p.is_file() and p.name!='sha256.json'}
    (dest/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(fits=len(fits),projections=len(projections),hashes=len(hashes),destination=str(dest))))


if __name__=='__main__':main()
