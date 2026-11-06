"""Fetch and verify an explicit remote inventory; emit a relocation receipt.

This local tool only downloads. It never removes remote files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/triangle-L6-m6-c8-D128-dense-20261106-v1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inventory',type=Path,required=True)
    args=parser.parse_args()
    inventory=json.loads(args.inventory.read_text())
    expected={r['path']:r for r in inventory['files']}
    env=dict(os.environ)
    if not env.get('SSHPASS'):
        env['SSHPASS']=subprocess.check_output(['security','find-generic-password','-s','autodl-password','-w'],text=True).strip()
    assert env['SSHPASS']
    receipt=[];total=0;pending=[]
    for row in inventory['files']:
        relative=Path(row['path'])
        category='notebook' if relative.suffix=='.ipynb' else 'data/kagome'
        target=ROOT/category/'autodl_relocated_for_D128_20261106'/relative
        if target.exists() and target.stat().st_size==row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']:
            receipt.append(dict(row,local_path=str(target)));total+=row['bytes']
        else:pending.append(row['path'])
    print('RESUME',len(receipt),'verified files;',len(pending),'remaining',flush=True)
    script="""from pathlib import Path
import json,tarfile,sys
base=Path('/root/autodl-tmp')
plan=json.loads(Path('%s/storage_relocation_inventory.json').read_text())
pending=set(%s)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for row in plan['files']:
  if row['path'] in pending:archive.add(base/row['path'],arcname=row['path'],recursive=False)
""" % (REMOTE,repr(pending))
    proc=subprocess.Popen(['sshpass','-e','ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','autodl',
                           '/root/miniconda3/bin/python -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env)
    proc.stdin.write(script.encode());proc.stdin.close()
    with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected
            relative=Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            category='notebook' if relative.suffix=='.ipynb' else 'data/kagome'
            target=ROOT/category/'autodl_relocated_for_D128_20261106'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.download')
            h=hashlib.sha256()
            with archive.extractfile(member) as src,partial.open('wb') as dst:
                for chunk in iter(lambda:src.read(4*2**20),b''):
                    h.update(chunk);dst.write(chunk)
            row=expected[member.name]
            assert partial.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'], member.name
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
            partial.replace(target)
            receipt.append(dict(row,local_path=str(target)))
            total+=row['bytes']
            if len(receipt)%500==0:print('VERIFIED',len(receipt),'files',round(total/2**30,3),'GiB',flush=True)
    assert proc.wait()==0 and len(receipt)==len(expected)
    path=args.inventory.with_name('storage_relocation_receipt.json')
    path.write_text(json.dumps(dict(files=receipt,total_bytes=total),indent=2)+'\n')
    print('VERIFIED_ALL',len(receipt),'files',total/2**30,'GiB',path,flush=True)


if __name__=='__main__':
    main()
