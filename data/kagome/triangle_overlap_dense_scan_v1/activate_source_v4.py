"""Adopt tested reclaimable-cache accounting without interrupting workers."""
import json
import os
from pathlib import Path
import subprocess
import time

from launch_handover import BASE, DENSE, ORIGINAL, PYTHON, stop_controller, command


def spawn(previous, log):
    env = dict(os.environ, PYTHONPATH=str(BASE/'triangle-correlation-scaling-20261104-v1/vendor')+':'+str(DENSE/'source-v4/code'),
               **{name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    with log.open('ab') as stream:
        proc = subprocess.Popen(previous['argv'], env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=stream, start_new_session=True)
    return dict(pid=proc.pid, argv=previous['argv'], log=str(log))


def main():
    evidence = DENSE/'handover-v4'
    evidence.mkdir(exist_ok=False)
    assert json.loads((DENSE/'preflight-v4/passed.json').read_text())['passed']
    prior = json.loads((DENSE/'handover-v3/launch.json').read_text())
    dense_status = json.loads((DENSE/'results/status.json').read_text())
    assert not dense_status['running'], 'Do not migrate unrecorded dense workers'
    before = {pid:command(pid) for pid in prior['preserved_workers']}
    assert all(before.values())
    stop_controller(prior['original']['pid'], 'transformer_qmc.triangle_correlation_parallel', ORIGINAL)
    stop_controller(prior['dense']['pid'], 'transformer_qmc.triangle_overlap_dense', DENSE)
    # Preserve the immutable v3 policy; v4 writes a new policy with its source hashes.
    (DENSE/'results/execution_manifest.json').rename(evidence/'execution_manifest_before.json')
    (DENSE/'results/status.json').rename(evidence/'dense_status_before.json')
    dense = spawn(prior['dense'], evidence/'dense-controller.log')
    # Give the prioritized requested validation the first admission opportunity.
    for _ in range(100):
        path = DENSE/'results/status.json'
        if path.exists():
            status = json.loads(path.read_text())
            if status.get('running'):
                break
        time.sleep(.2)
    original = spawn(prior['original'], evidence/'original-controller.log')
    assert all(command(pid)==argv for pid,argv in before.items())
    launch = dict(prior, utc=time.time(), original=original, dense=dense,
                  source=str(DENSE/'source-v4'),
                  memory_accounting='unused plus reclaimable inactive file pages; active file, shmem, anon remain charged',
                  supersedes=str(DENSE/'handover-v3/launch.json'))
    (evidence/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2),flush=True)


if __name__=='__main__':
    main()
