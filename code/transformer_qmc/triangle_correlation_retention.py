"""Finite L8 m/chi study with archived inputs and unchanged solver gates."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_correlation_dmrg import case, digest, write
from .triangle_correlation_observables import ratio

FIELDS = [.1, .5, 1.]
MS = [4, 6, 8]
CHIS = [2, 3, 4]
IDS = [60000, 60001]


def endpoint(row):
    return min((next(r for r in reversed(row['records']) if r['seed'] == seed)
                for seed in (0, 1)), key=lambda r: r['energy'])


def report(root):
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*/result.json')]
    passed = [r for r in rows if r['L'] == 8 and r['solver_passed']]
    lookup = {(r['gamma'], r['sample_id'], r['m'], r['chi']): r for r in passed}
    means = []
    for g in FIELDS:
        for m in MS:
            for chi in CHIS:
                if all((g, s, m, chi) in lookup for s in IDS):
                    spectra = np.array([endpoint(lookup[g, s, m, chi])['spectra'] for s in IDS])
                    rr = float(ratio(spectra.mean(axis=0)))
                    means.append(dict(gamma=g, m=m, chi=chi, R=rr if np.isfinite(rr) else None,
                                      spectra=spectra.mean(axis=0).tolist()))
    destination = root/'report'
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
    for i, g in enumerate(FIELDS):
        for j in range(3):
            ax = axes[i, j]
            for m, color in zip(MS, ('#0072B2', '#D55E00', '#009E73')):
                values = []
                for chi in CHIS:
                    if j < 2:
                        row = lookup.get((g, IDS[j], m, chi))
                        rr = endpoint(row)['R'] if row else None
                    else:
                        rr = next((r['R'] for r in means if (r['gamma'], r['m'], r['chi']) == (g, m, chi)), None)
                    values.append(rr if rr is not None else np.nan)
                ax.plot(CHIS, values, 'o-', color=color, label=f'm={m}')
            label = f'disorder {IDS[j]}' if j < 2 else 'ratio of two-disorder mean spectra'
            ax.set(title=f'Gamma={g:g} | {label}', xlabel='chi', ylabel='R = xi_R / L', xticks=CHIS)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('L8 retention study | solver-passing points only | no error bars')
    fig.savefig(destination/'L8_retention.png', dpi=160)
    fig.savefig(destination/'L8_retention.pdf')
    plt.close(fig)
    status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    failures = [{k:r[k] for k in ('L','gamma','sample_id','m','chi')} for r in rows if not r['solver_passed']]
    summary = dict(status=status, solver_passing_L8=len(passed), target_L8=54,
                   two_disorder_means=means, unresolved_solver_cases=failures,
                   interpretation='Two-disorder sensitivity study; physical retention convergence unestablished')
    write(destination/'summary.json', summary)
    image = base64.b64encode((destination/'L8_retention.png').read_bytes()).decode()
    cells = [nbformat.v4.new_markdown_cell('# L8: sensitivity to m and chi\n\n'
        'Gamma=0.1,0.5,1.0; m=4,6,8; chi=2,3,4; matched disorders 60000,60001. '
        'Original K~Uniform(0.5,1.5), lambda=1, h=0.15. '
        'm is states retained per triangle (m=8 is complete); chi is states retained per 12-spin superblock. '
        'D_MPS separately controls the effective-Hamiltonian solve. '
        f'{len(passed)}/54 L8 cases pass the original solver gates. '
        'Dense Gamma scan is held. Missing points are unfinished or unresolved. '
        'Lines only join adjacent tested chi values. No error bars for this two-disorder pilot.'),
        nbformat.v4.new_markdown_cell('![L8 retention](attachment:L8_retention.png)',
            attachments={'L8_retention.png': {'image/png':image}}),
        nbformat.v4.new_markdown_cell('## Status\n\n```json\n'+json.dumps(status,indent=2)+'\n```'),
        nbformat.v4.new_markdown_cell('## Interpretation\n\nCompare m at fixed chi and chi at fixed m. '
            'Inspect each disorder as well as the ratio of averaged spectra. A systematic plateau shift with '
            'retention indicates truncation sensitivity. Agreement across this small ladder is evidence '
            'of stability within these settings, not a physical convergence proof. '
            'The former 0.01 criterion for R is descriptive here; two disorders do not qualify production curves. '
            'New settings must pass exact reduced-space L4 checks before L8; solver failures are preserved '
            'and excluded from physical comparisons.'),
        nbformat.v4.new_markdown_cell('## Unresolved solver cases\n\n```json\n'+json.dumps(failures,indent=2)+'\n```')]
    nb = nbformat.v4.new_notebook(cells=cells, metadata=dict(execution_location='AutoDL',
        kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb)
    nbformat.write(nb, destination/'triangle_correlation_L8_retention.ipynb')


def reuse(root, previous):
    records = []
    for folder in sorted((previous/'cases').iterdir()):
        path = folder/'result.json'
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row['L'] in (4,8) and row['sample_id'] in IDS:
            for name, value in row['hashes'].items():
                assert digest(folder/name) == value
            target = root/'cases'/folder.name
            if not target.exists():
                shutil.copytree(folder,target,copy_function=os.link)
            records.append(dict(source=str(folder), result_sha256=digest(path)))
    write(root/'reuse.json', records)


def import_diagnostic(root, diagnostic):
    assessment = diagnostic/'assessment.json'
    if not assessment.exists():
        return
    value = json.loads(assessment.read_text())
    write(root/'diagnostic_assessment.json',value)
    if not value['solver_passed'] or set(value['seeds']) != {0,1}:
        return
    folder = root/'cases/L8_g1_s60000_m6_c3'
    path = folder/'result.json'
    original = json.loads(path.read_text())
    if original['solver_passed']:
        return
    original_sha = digest(path)
    rows = []
    for seed in (0,1):
        row = json.loads((diagnostic/f'seed{seed}_D1024.json').read_text())
        assert row['ladder_stable'] and row['original_result_sha256'] == original_sha
        rows.append(row)
        for suffix in ('.pkl','.npz','.json'):
            name = f'seed{seed}_D1024{suffix}'
            shutil.copy2(diagnostic/name,folder/name)
            if suffix != '.json':
                original['hashes'][name] = digest(folder/name)
    original['records'].extend(rows)
    original['solver_passed'] = True
    original['continued_from_sha256'] = original_sha
    original['continuation_assessment_sha256'] = digest(assessment)
    write(path,original)  # Atomic replacement preserves the previous hard-linked result.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--diagnostic',type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    (root/'cases').mkdir(exist_ok=True)
    manifest = dict(L=8,gammas=FIELDS,m=MS,chi=CHIS,disorders=IDS,seeds=[0,1],
        D_MPS=[32,64,128,256,512],worker_threads=1,
        sources={str(p.relative_to(Path(__file__).parent)):digest(p) for p in Path(__file__).parent.glob('*.py')})
    if (root/'manifest.json').exists():
        assert json.loads((root/'manifest.json').read_text()) == manifest
    else:
        write(root/'manifest.json',manifest)
    reuse(root,args.previous)
    write(root/'status.json',dict(state='waiting_for_diagnostic' if args.wait_pid else 'running',wait_pid=args.wait_pid))
    report(root)
    if args.wait_pid:
        while True:
            try:
                os.kill(args.wait_pid,0)
            except ProcessLookupError:
                break
            time.sleep(30)
    if args.diagnostic:
        import_diagnostic(root,args.diagnostic)
        report(root)
    try:
        for m in MS:
            for chi in CHIS:
                write(root/'status.json',dict(state='running',phase='L4_validation',m=m,chi=chi))
                validation = case(root,4,.5,60000,chi,validate=True,m=m)
                report(root)
                if not validation['solver_passed']:
                    continue
                for g in FIELDS:
                    for s in IDS:
                        if shutil.disk_usage(root).free < 3*2**30:
                            raise RuntimeError('Less than 3 GiB disk headroom; preserve outputs and archive before continuing')
                        write(root/'status.json',dict(state='running',phase='L8',m=m,chi=chi,gamma=g,sample_id=s))
                        case(root,8,g,s,chi,m=m)
                        report(root)
        rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*/result.json')]
        passed=sum(r['L']==8 and r['solver_passed'] for r in rows)
        write(root/'status.json',dict(state='complete' if passed==54 else 'complete_with_unresolved_cases',
              solver_passing_L8=passed,target_L8=54,dense_scan='held'))
    except Exception:
        (root/'failure.txt').write_text(traceback.format_exc())
        write(root/'status.json',dict(state='blocked',error=traceback.format_exc(),dense_scan='held'))
        raise
    finally:
        report(root)


if __name__ == '__main__':
    main()
