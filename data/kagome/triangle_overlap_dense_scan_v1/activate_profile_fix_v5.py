"""Explicitly rerun one failed timing profile; adopt active scientific workers."""
import json
import time

from launch_handover import DENSE, stop_controller, command


def main():
    evidence = DENSE/'handover-v5'
    evidence.mkdir(exist_ok=False)
    assert json.loads((DENSE/'preflight-v5/passed.json').read_text())['passed']
    prior = json.loads((DENSE/'handover-v4/launch.json').read_text())
    stop_controller(prior['dense']['pid'], 'transformer_qmc.triangle_overlap_dense', DENSE)
    status = json.loads((DENSE/'results/status.json').read_text())
    live = {j['pid']:command(j['pid']) for j in status['running'].values() if command(j['pid'])}
    (evidence/'dense_status_before.json').write_text(json.dumps(status,indent=2)+'\n')
    ledger_path = DENSE/'results/attempts.json'
    ledger = json.loads(ledger_path.read_text())
    (evidence/'attempts_before.json').write_text(json.dumps(ledger,indent=2)+'\n')
    name = 'profile_L6_g0.5_s60000_m8_c16'
    failed = ledger.pop(name)
    assert failed['outcome']=='unresolved' and 'Maximum truncation error' in failed['exit']['failure']
    for relative in ['profiles/L6_g0.5_s60000_m8_c16', f'logs/{name}.log', f'logs/{name}.exit.json']:
        source = DENSE/'results'/relative
        source.rename(evidence/source.name)
    temporary = ledger_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(ledger,indent=2)+'\n')
    temporary.replace(ledger_path)
    (DENSE/'results/execution_manifest.json').rename(evidence/'execution_manifest_before.json')
    import os, subprocess
    env = dict(os.environ, PYTHONPATH='/root/autodl-tmp/triangle-correlation-scaling-20261104-v1/vendor:'+str(DENSE/'source-v5/code'),
               **{n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    log = evidence/'dense-controller.log'
    with log.open('ab') as stream:
        proc = subprocess.Popen(prior['dense']['argv'],env=env,stdin=subprocess.DEVNULL,
                                stdout=stream,stderr=stream,start_new_session=True)
    assert all(command(pid)==argv for pid,argv in live.items())
    launch = dict(prior,utc=time.time(),dense=dict(pid=proc.pid,argv=prior['dense']['argv'],log=str(log)),
                  source=str(DENSE/'source-v5'), original_source=str(DENSE/'source-v4'),
                  preserved_dense_workers=list(live), supersedes=str(DENSE/'handover-v4/launch.json'),
                  explicit_profile_retry=dict(name=name,reason='Record D16 timing truncation as a diagnostic; production solver gates unchanged',
                                              archived_failure=str(evidence)))
    (evidence/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2),flush=True)


if __name__=='__main__':
    main()
