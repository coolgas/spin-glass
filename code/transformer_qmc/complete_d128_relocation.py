"""Complete an explicitly inventoried relocation after local hash verification.

Only byte-identical remote files with a durable local counterpart are unlinked.
Directories, unlisted files, changed files, and active process references survive.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/kagome/triangle_L6_m6_chi8_D128_diagnostics_20261106'
REMOTE='/root/autodl-tmp/triangle-L6-m6-c8-D128-dense-20261106-v1'


def main():
    receipt_path=DATA/'storage_relocation_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    inventory=json.loads((DATA/'storage_relocation_inventory.json').read_text())
    assert {r['path']:(r['bytes'],r['sha256']) for r in receipt['files']} == {
        r['path']:(r['bytes'],r['sha256']) for r in inventory['files']}
    for row in receipt['files']:
        p=Path(row['local_path'])
        assert p.is_relative_to(ROOT) and p.stat().st_size==row['bytes']
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*2**20),b''):h.update(chunk)
        assert h.hexdigest()==row['sha256'],str(p)
    env=dict(os.environ)
    if not env.get('SSHPASS'):
        env['SSHPASS']=subprocess.check_output(['security','find-generic-password','-s','autodl-password','-w'],text=True).strip()
    assert env['SSHPASS']
    subprocess.run(['sshpass','-e','scp','-o','ConnectTimeout=15',str(receipt_path),
                    'autodl:'+REMOTE+'/storage_relocation_receipt.json'],env=env,check=True)
    script=r'''from pathlib import Path
import json,hashlib,os,shutil,time
root=Path('/root/autodl-tmp/triangle-L6-m6-c8-D128-dense-20261106-v1')
base=Path('/root/autodl-tmp')
receipt=json.loads((root/'storage_relocation_receipt.json').read_text())
inventory=json.loads((root/'storage_relocation_inventory.json').read_text())
assert {r['path']:(r['bytes'],r['sha256']) for r in receipt['files']} == {r['path']:(r['bytes'],r['sha256']) for r in inventory['files']}
refs=[]
for proc in Path('/proc').glob('[0-9]*'):
 try:
  refs += [str((proc/'cwd').resolve())]+[str(p.resolve()) for p in (proc/'fd').iterdir()]+[(proc/'cmdline').read_bytes().decode(errors='replace')]
 except OSError:pass
for directory in inventory['directories']:
 prefix=str(base/directory)
 assert not any(prefix+'/' in x or prefix+'\x00' in x or prefix==x for x in refs), prefix
verified=[]
for row in receipt['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 p=base/relative
 if not p.exists():continue
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'],str(p)
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(4*2**20),b''):h.update(chunk)
 assert h.hexdigest()==row['sha256'],str(p)
 verified.append((p,p.stat().st_mtime_ns,row))
before=shutil.disk_usage(base).free
for p,mtime,row in verified:
 assert p.stat().st_mtime_ns==mtime and p.stat().st_size==row['bytes']
 p.unlink()
for directory in inventory['directories']:
 p=base/directory/'RELOCATED_TO_LOCAL_20261106.json'
 p.write_text(json.dumps(dict(receipt=str(root/'storage_relocation_receipt.json'),
     local_data='/Users/tianyiyan/Dropbox/spin-glasses/data/kagome/autodl_relocated_for_D128_20261106',
     local_notebooks='/Users/tianyiyan/Dropbox/spin-glasses/notebook/autodl_relocated_for_D128_20261106',
     note='Only inventoried, hash-verified files relocated; unlisted files and directories retained.'),indent=2)+'\n')
result=dict(utc=time.time(),files=len(verified),bytes=sum(r['bytes'] for _,_,r in verified),
            before_free_gib=before/2**30,after_free_gib=shutil.disk_usage(base).free/2**30)
(root/'storage_relocation_complete.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
'''
    subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl','/root/miniconda3/bin/python -'],
                   input=script.encode(),env=env,check=True)
    subprocess.run(['sshpass','-e','scp','-o','ConnectTimeout=15',
                    'autodl:'+REMOTE+'/storage_relocation_complete.json',str(DATA)],env=env,check=True)


if __name__=='__main__':main()
