"""Wait for a detached campaign, then assess complete groups; never promote runs."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root
    launch=json.loads((root/'launch.json').read_text());pid=launch['pid']
    while True:
        status=root/'results/gfmc_status.json'
        if status.exists() and json.loads(status.read_text())['state']=='completed':
            state='completed';break
        proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'transformer_qmc.correlation_campaign' not in proc.read_bytes():
            state='stopped_before_completion';break
        time.sleep(30)
    sys.argv=['assess',str(root/'results')]
    module=runpy.run_path(str(root/'assess_correlation_campaign.py'),run_name='transformer_qmc._assessment')
    module['main']()
    (root/'review_status.json').write_text(json.dumps(dict(campaign_state=state,reviewed_unix=time.time()))+'\n')
    print(state,flush=True)


if __name__=='__main__':main()
