"""Audited AutoDL controller handover; preserve every active scientific worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BASE = Path('/root/autodl-tmp')
DENSE = BASE/'triangle-overlap-dense-scan-20261105-v1'
ORIGINAL = BASE/'triangle-correlation-retention-20261105-v6'
EXTENSION = BASE/'triangle-correlation-highchi-20261105-v7'
PYTHON = '/root/miniconda3/bin/python'
LOCK = BASE/'triangle-correlation-shared-dispatch.lock'


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:
        return []


def stop_controller(pid, module, root):
    argv = command(pid)
    assert module in argv and str(root/'results') in argv and '--task-json' not in argv, argv
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not command(pid):
            return
        time.sleep(.1)
    raise RuntimeError(f'Controller {pid} did not exit')


def spawn(module, arguments, log):
    env = dict(os.environ, PYTHONPATH=str(BASE/'triangle-correlation-scaling-20261104-v1/vendor')+':'+str(DENSE/'source-v3/code'),
               **{name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    argv = [PYTHON, '-u', '-m', module, *map(str, arguments)]
    with log.open('ab') as stream:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=stream, start_new_session=True)
    return dict(pid=proc.pid, argv=argv, log=str(log))


def main():
    evidence = DENSE/'handover-v3'
    evidence.mkdir(exist_ok=False)
    assert json.loads((DENSE/'preflight-v3/passed.json').read_text())['passed']
    previous = json.loads((ORIGINAL/'results/status.json').read_text())
    extension = json.loads((EXTENSION/'results/status.json').read_text())
    assert extension['state']=='waiting_for_predecessors' and not extension['running']
    before = {str(j['pid']): command(j['pid']) for j in previous['running']}
    assert all(before.values())
    (evidence/'before.json').write_text(json.dumps(dict(original=previous, extension=extension, workers=before),indent=2)+'\n')
    stop_controller(540660, 'transformer_qmc.triangle_correlation_highchi', EXTENSION)
    held = dict(extension, state='held_for_requested_dense_campaign', held_utc=time.time(),
                stopped_controller_pid=540660, reason='Prioritize authorized L4/L6/L8 m8 chi3/16 scan; preserve broad-extension inputs.')
    (EXTENSION/'results/status.json').write_text(json.dumps(held,indent=2)+'\n')
    stop_controller(521299, 'transformer_qmc.triangle_correlation_parallel', ORIGINAL)
    assert all(command(int(pid))==argv for pid,argv in before.items())
    original = spawn('transformer_qmc.triangle_correlation_parallel', [
        '--output',ORIGINAL/'results','--previous',BASE/'triangle-correlation-retention-20261105-v5/results',
        '--wait-pid',515820,'--diagnostic',BASE/'triangle-correlation-scaling-20261105-v4/diagnostic',
        '--workers',16,'--threads',1,'--memory-budget-gib',48,
        '--peer-roots',DENSE/'results','--dispatch-lock',LOCK], evidence/'original-controller.log')
    (evidence/'original_launch.json').write_text(json.dumps(original,indent=2)+'\n')
    dense = spawn('transformer_qmc.triangle_overlap_dense', [
        '--output',DENSE/'results','--reuse',ORIGINAL/'results',
        '--backend',EXTENSION/'results/backend_benchmark.json',
        '--profile-threads',2,'--dispatch-lock',LOCK], evidence/'dense-controller.log')
    launch = dict(utc=time.time(), original=original, dense=dense, preserved_workers=list(map(int,before)),
                  held_controller=540660, source=str(DENSE/'source-v3'), dispatch_lock=str(LOCK))
    (evidence/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2),flush=True)


if __name__=='__main__':
    main()
