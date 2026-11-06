"""One-shot evidence download and execution of the local D128 notebook."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

import nbformat
from nbclient import NotebookClient

from .build_l6_d128_notebook import main as build
from .l6_d128_diagnostics import DATA128

ROOT=Path(__file__).resolve().parents[2]


def main():
    data=ROOT/'data/kagome'/DATA128
    raw=data/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    if not env.get('SSHPASS'):
        env['SSHPASS']=subprocess.check_output(['security','find-generic-password','-s','autodl-password','-w'],text=True).strip()
    assert env['SSHPASS']
    script="""from pathlib import Path
import json,tarfile,sys
r=Path('/root/autodl-tmp/triangle-L6-m6-c8-D128-dense-20261106-v1')
paths=list((r/'results').glob('*.json'))+list((r/'results/attempts').glob('*.json'))
paths += [p for p in r.glob('*.json') if not p.name.startswith('storage_relocation')]
for p in (r/'results/cases').glob('*/result.json'):
 paths.extend([p,p.parent/'ancestry.json'])
 paths.extend(p.parent.glob('seed*_D128.json'))
 paths.extend(p.parent.glob('seed*_D128.npz'))
for filename in ['triangle_l6_d128.py','l6_d64_diagnostics.py','triangle_effective_mps.py','triangle_effective_rg.py','triangle_correlation_dmrg.py']:
 paths.append(r/'source/code/transformer_qmc'/filename)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for p in sorted(set(paths)):
  if p.is_file():archive.add(p,arcname=str(p.relative_to(r)),recursive=False)
"""
    partial=data/'evidence.tar.gz.download'
    with partial.open('wb') as stream:
        subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl','/root/miniconda3/bin/python -'],
                       input=script.encode(),stdout=stream,env=env,check=True)
    partial.replace(data/'evidence.tar.gz')
    with tarfile.open(data/'evidence.tar.gz') as archive:
        for member in archive.getmembers():
            p=Path(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts and p.suffix!='.ipynb'
        archive.extractall(raw,filter='data')
    build()
    target=ROOT/'notebook/triangle_L6_m6_chi8_D128_convergence.ipynb'
    notebook=nbformat.read(target,as_version=4)
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(target.parent)}}).execute()
    nbformat.validate(notebook);nbformat.write(notebook,target)
    print(json.loads((data/'summary.json').read_text()))
    print('EXECUTED',target,'figures',sum('image/png' in o.get('data',{}) for c in notebook.cells for o in c.get('outputs',[])))


if __name__=='__main__':main()
