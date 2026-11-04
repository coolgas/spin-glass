"""Verify the downloaded L6 field scan and install its embedded notebook."""
import base64
import hashlib
import json
from pathlib import Path
import shutil

import nbformat
import numpy as np


def verify(root):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    results=root/'results'
    status=json.loads((results/'status.json').read_text())
    assert status['state']=='complete' and status['completed']==120 and not status['failures']
    for manifest,source in (('results/launch.json','code'),('reused_launch.json','reused_source'),
                            ('original_reused_launch.json','original_reused_source')):
        for name,value in json.loads((root/manifest).read_text())['sources'].items():
            assert sha(root/source/name)==value,(source,name)
    rows=[json.loads(p.read_text()) for p in (results/'cases').glob('*.json')]
    assert len(rows)==120
    bonds={};energy_errors=[]
    for row in rows:
        assert row['L']==6 and row['n']==108 and row['h']==.15 and row['lambda_rg']==1
        path=results/'cases'/(row['key']+'.npz')
        assert sha(path)==row['raw_sha256']
        with np.load(path) as a:
            np.testing.assert_array_equal(a['J'],a['original_J'])
            assert np.all((a['J']>=-1.5)&(a['J']<=-.5))
            bh=hashlib.sha256(a['J'].tobytes()).hexdigest()
            assert bonds.setdefault(row['sample_id'],bh)==bh
            counts=np.bincount((a['spins'][::2]==a['spins'][1::2]).sum(1),minlength=109)
            np.testing.assert_array_equal(counts,a['counts'])
            np.testing.assert_array_equal(counts/4096,row['mass'])
            assert counts.sum()==4096
            np.testing.assert_allclose(a['state']@a['state'],1,atol=1e-11)
            z=a['magnetization'];zz=a['zz']
            np.testing.assert_allclose(z@z/108,row['metrics']['q'],atol=1e-12)
            np.testing.assert_allclose(np.sum(zz**2)/108**2,row['metrics']['q2'],atol=1e-12)
            energy=-sum(j*zz[i,k] for (i,k),j in zip(a['bonds'],a['J']))-.15*z.sum()-row['gamma']*a['xmag'].sum()
            energy_errors.append(abs(energy-row['energy']))
            assert energy_errors[-1]<1e-8
    summary=json.loads((results/'summary.json').read_text())
    assert summary['complete_samples']==list(range(60000,60008))
    lookup={(r['gamma'],r['sample_id'],r['m'],r['chi']):r for r in rows}
    def check(values,mean,se):
        v=np.asarray(values)
        np.testing.assert_allclose(v.mean(0),mean,atol=1e-13)
        np.testing.assert_allclose(v.std(0,ddof=1)/np.sqrt(8),se,atol=1e-13)
    def metrics(r):return np.array([r['metrics']['q'],r['metrics']['q2'],r['energy_per_spin']])
    for avg in summary['averages']:
        group=[lookup[avg['gamma'],s,avg['m'],avg['chi']] for s in summary['complete_samples']]
        check([r['mass'] for r in group],avg['mass'],avg['mass_disorder_se'])
        check([metrics(r) for r in group],avg['mean'],avg['se'])
    for axis in ('retention','field'):
        for d in summary['paired_'+axis+'_changes']:
            values=[]
            for s in summary['complete_samples']:
                if axis=='retention':
                    x,y=lookup[(d['fixed'],s)+tuple(d['source'])],lookup[(d['fixed'],s)+tuple(d['destination'])]
                else:
                    x,y=lookup[(d['source'],s)+tuple(d['fixed'])],lookup[(d['destination'],s)+tuple(d['fixed'])]
                values.append(metrics(y)-metrics(x))
            check(values,d['mean'],d['se'])
    prov=json.loads((results/'report_provenance.json').read_text())
    assert sha(root/'code/transformer_qmc/triangle_effective_l6_gamma_scan.py')==prov['source_sha256']
    for name,value in prov['files'].items():assert sha(results/name)==value,name
    notebook=results/'triangle_effective_l6_gamma_scan.ipynb'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    assert nb.metadata['error_bars']=='disorder SE only'
    assert len(nb.cells)==4
    for cell in nb.cells:
        for name,mimes in cell.get('attachments',{}).items():
            assert base64.b64decode(mimes['image/png'])==(results/name).read_bytes()
    target=root.parents[2]/'notebook'/notebook.name
    shutil.copy2(notebook,target)
    verification=dict(cases=120,raw_hashes_verified=120,source_snapshots_verified=True,
        matched_bonds_verified=True,histograms_verified=True,direct_moments_verified=True,
        disorder_averages_and_SE_verified=True,paired_changes_verified=True,
        report_hashes_verified=True,embedded_figures_verified=True,
        max_physical_energy_error=max(energy_errors),alarms=summary['alarms'],notebook_sha256=sha(target))
    (root/'local_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
    print('Installed:',target)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    verify(parser.parse_args().root.resolve())
