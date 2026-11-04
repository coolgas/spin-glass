"""Wait for the finite remote campaign, report, download, and verify delivery."""
import os
from pathlib import Path
import subprocess
import sys

import nbformat

REMOTE = '/root/autodl-tmp/triangle-rg-bond-ranges-20261104-v1'
WORKSPACE = Path(__file__).resolve().parents[3]
DELIVERY = Path(__file__).resolve().parent


def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    if not os.environ.get('SSHPASS'):
        os.environ['SSHPASS'] = subprocess.check_output(
            ['security','find-generic-password','-s','autodl-password','-w'],text=True).strip()
    remote_script = r'''
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,tarfile
r=Path('/root/autodl-tmp/triangle-rg-bond-ranges-20261104-v1')
os.chdir(r)
previous=None
while True:
    status=json.loads((r/'results/status.json').read_text())
    if status!=previous:
        print(json.dumps(status),flush=True)
        previous=status
    if status['state']!='running': break
    time.sleep(30)
assert status['state']=='complete',status
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='code')
with (r/'report.log').open('w') as log:
    subprocess.run(['/root/miniconda3/bin/python','-u','-m','transformer_qmc.triangle_effective_bond_report','--output','results'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
(r/'report_sources').mkdir(exist_ok=True)
for name in ['triangle_effective_bond_scan.py','triangle_effective_bond_report.py']:
    shutil.copy2(r/'code/transformer_qmc'/name,r/'report_sources'/name)
launch=json.loads((r/'results/launch.json').read_text())
with tarfile.open(r/'source_snapshot.tar.gz') as archive:
    for path,expected in launch['sources'].items():
        assert hashlib.sha256(archive.extractfile('code/'+path).read()).hexdigest()==expected,path
print('Launch-source archive verified; saved-data report complete.',flush=True)
'''
    run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl',
         '/root/miniconda3/bin/python -u -'],input=remote_script,text=True)
    # Deliver reviewable reports before the larger raw archives.
    run(['rsync','-az','-e','sshpass -e ssh',
         '--include=results/','--include=results/*.png','--include=results/*.pdf',
         '--include=results/*.ipynb','--include=results/*.json','--include=results/*.md',
         '--include=report_sources/','--include=report_sources/***','--include=report.log',
         '--include=source_snapshot.tar.gz','--exclude=*',
         f'autodl:{REMOTE}/',str(DELIVERY)+'/'])
    notebook = DELIVERY/'results/triangle_effective_bond_scan.ipynb'
    nbformat.validate(nbformat.read(notebook,as_version=4))
    import shutil
    shutil.copy2(notebook,WORKSPACE/'notebook/triangle_effective_bond_scan.ipynb')
    print('Report notebook delivered; downloading raw archives for local verification.',flush=True)
    run(['rsync','-az','-e','sshpass -e ssh',
         '--include=results/','--include=results/***','--include=report_sources/',
         '--include=report_sources/***','--include=source_snapshot.tar.gz',
         '--include=campaign.log','--include=report.log',
         '--include=pilot_unrestricted_results/','--include=pilot_unrestricted_results/***',
         '--exclude=*',f'autodl:{REMOTE}/',str(DELIVERY)+'/'])
    run([sys.executable,str(WORKSPACE/'code/transformer_qmc/verify_triangle_bond_delivery.py'),
         '--delivery',str(DELIVERY),'--notebook',str(WORKSPACE/'notebook/triangle_effective_bond_scan.ipynb')])
    print('All-bond scan delivery and local verification complete.',flush=True)


if __name__=='__main__':
    main()
