"""Gate the next campaign on a separately archived D1024 diagnosis."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .triangle_correlation_dmrg import digest, write
from .triangle_correlation_report import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--diagnostic-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    while not (args.diagnostic/'assessment.json').exists():
        try:
            os.kill(args.diagnostic_pid, 0)
        except ProcessLookupError:
            write(target/'resume.json', dict(state='blocked', reason='Diagnostic exited without assessment'))
            return
        time.sleep(30)
    assessment = json.loads((args.diagnostic/'assessment.json').read_text())
    if not assessment['solver_passed'] or set(assessment['seeds']) != {0, 1}:
        write(target/'resume.json', dict(state='blocked', reason='D1024 did not meet the original ladder gates',
                                         diagnostic=assessment))
        return
    source = args.previous/'cases/L8_g1_s60000_m6_c3'
    original = json.loads((source/'result.json').read_text())
    for filename, expected in original['hashes'].items():
        assert digest(source/filename) == expected
    target_cases = target/'cases'
    target_cases.mkdir(exist_ok=True)
    copied=[]
    for folder in sorted((args.previous/'cases').iterdir()):
        file = folder/'result.json'
        if file.exists() and json.loads(file.read_text())['solver_passed']:
            shutil.copytree(folder, target_cases/folder.name, copy_function=os.link, dirs_exist_ok=True)
            copied.append(dict(case=folder.name, original_sha256=digest(file)))
    repaired=target_cases/source.name
    shutil.copytree(source, repaired, copy_function=os.link, dirs_exist_ok=True)
    rows=[]
    for seed in (0,1):
        row=json.loads((args.diagnostic/f'seed{seed}_D1024.json').read_text())
        assert row['ladder_stable'] and row['original_result_sha256'] == digest(source/'result.json')
        rows.append(row)
        for suffix in ('.pkl','.npz'):
            name=f'seed{seed}_D1024{suffix}'
            shutil.copy2(args.diagnostic/name, repaired/name)
        write(repaired/f'seed{seed}_D1024.json',row)
    endpoints=[r for r in rows]
    assert abs(endpoints[0]['energy']-endpoints[1]['energy'])/192 < 1e-6
    assert abs(endpoints[0]['R']-endpoints[1]['R']) < .002
    assert np.max(abs(np.array(endpoints[0]['spectra'])-endpoints[1]['spectra'])/
                  np.maximum(abs(endpoints[0]['spectra']),1e-10)) < .01
    original['records'].extend(rows)
    original['solver_passed']=True
    original['continued_from_sha256']=digest(source/'result.json')
    original['continuation_assessment_sha256']=digest(args.diagnostic/'assessment.json')
    original['hashes'].update({p.name:digest(p) for p in repaired.glob('*D1024.*') if p.suffix in ('.pkl','.npz')})
    write(repaired/'result.json',original)
    write(target/'reuse.json',dict(copied=copied,repaired_source=str(source),
                                   repaired_original_sha256=digest(source/'result.json'),
                                   diagnostic=str(args.diagnostic),assessment_sha256=digest(args.diagnostic/'assessment.json')))
    report(target)
    for phase in ('frontier','scan'):
        write(target/'resume.json',dict(state='running',phase=phase))
        with (target/f'{phase}.log').open('ab') as stream:
            process=subprocess.run([sys.executable,'-u','-m','transformer_qmc.triangle_correlation_dmrg',
                                    '--output',str(target),'--phase',phase],stdout=stream,stderr=subprocess.STDOUT)
        report(target)
        if process.returncode:
            write(target/'resume.json',dict(state='blocked',phase=phase,
                status=json.loads((target/'status.json').read_text())))
            return
        write(target/'resume.json',dict(state='complete',phase=phase))


if __name__=='__main__':
    main()
