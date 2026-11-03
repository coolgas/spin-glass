"""Bounded AutoDL monitor and offline assessment for the worldline development batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--remote',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];dest=root/'data/kagome/ltfim_worldline_v1';dest.mkdir(parents=True,exist_ok=True)
    script="""
from pathlib import Path
import json
root=Path(REMOTE)
p=root/'results/status.json'
status=json.loads(p.read_text()) if p.exists() else {'state':'starting','completed':0}
pid=json.loads((root/'launch.json').read_text())['pid']
p=Path('/proc')/str(pid)/'cmdline'
status['alive']=p.exists() and b'transformer_qmc.worldline_campaign' in p.read_bytes()
if not status['alive'] and status['state']!='completed':status['log_tail']=(root/'campaign.log').read_text()[-6000:]
print(json.dumps(status))
""".replace('REMOTE',repr(a.remote))
    deadline=time.monotonic()+8*3600;previous=None
    while time.monotonic()<deadline:
        r=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl','/root/miniconda3/bin/python -'],
            input=script,text=True,capture_output=True,timeout=45)
        if r.returncode:raise RuntimeError(r.stderr)
        status=json.loads(r.stdout)
        if status!=previous:print(json.dumps(status),flush=True);previous=status
        (dest/'remote_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if status['state']=='completed' or not status['alive']:
            subprocess.run(['rsync','-az','-e','sshpass -e ssh -o ConnectTimeout=15',
                f'autodl:{a.remote}/results/',str(dest)+'/'],check=True,timeout=600)
            if status['state']!='completed':raise RuntimeError(status)
            subprocess.run([sys.executable,'-m','transformer_qmc.assess_worldline',str(dest),'--references',
                str(root/'data/kagome/ltfim_correlation_expanded_v1/guides')],cwd=root,check=True,timeout=600)
            return
        time.sleep(30)
    raise TimeoutError('Worldline monitor deadline reached; remote job left running')


if __name__=='__main__':main()
