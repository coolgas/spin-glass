"""Review complete L2 convergence sets, separating MC bias from disorder error."""
import argparse
import json
from pathlib import Path

import numpy as np

from .correlation_lengths import length


def name(sid, gamma):
    return f'L2_s{sid}_g{gamma:g}_h0p15'.replace('.', 'p')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root/'manifest.json').read_text())
    ids = manifest['development_ids']
    draws = 4000
    rng = np.random.default_rng(20261103)
    rows, screens, incomplete = [], [], []
    for gamma in manifest['convergence_gamma']:
        exact = np.array([json.loads((root/'exact'/(name(s,gamma)+'.json')).read_text())['spectrum'] for s in ids])
        target = exact.mean(0)
        target_x = float(length(target,2)/4)
        for setting in manifest['settings']:
            paths = [[root/'gfmc'/(name(s,gamma)+f'_{setting}_r{r}.json') for r in range(4)] for s in ids]
            missing = sum(not p.exists() for group in paths for p in group)
            if missing:
                incomplete.append(dict(gamma=gamma,setting=setting,missing=missing))
                continue
            records = [[json.loads(p.read_text()) for p in group] for group in paths]
            runs = np.array([[r['result']['connected_spectrum'] for r in group] for group in records])
            mean = runs.mean(axis=(0,1))
            # Fixed development disorders: resample repeats only to test numerical bias.
            repeats = rng.integers(4,size=(draws,len(ids),4))
            boot = runs[np.arange(len(ids))[None,:,None],repeats].mean(axis=(1,2))
            xb = length(boot,2)/4
            x = float(length(mean,2)/4)
            spectral_se = boot.std(0,ddof=1)
            spectral_bound = (np.abs(mean-target)+1.96*spectral_se)/target
            invalid = float(np.mean(~np.isfinite(xb)))
            finite = xb[np.isfinite(xb)]
            xse = float(finite.std(ddof=1)) if len(finite)>1 else None
            bound = abs(x-target_x)+1.96*xse if np.isfinite(x) and xse is not None else None
            alarms = sum(r['ancestry_alarm'] for group in records for r in group)
            passes = (bound is not None and bound <= .01 and invalid == 0 and
                      np.all(spectral_bound <= .05) and alarms == 0)
            rows.append(dict(gamma=gamma,setting=setting,disorders=len(ids),repeats=4,
                             xi=float(x*4) if np.isfinite(x) else None,exact_xi=target_x*4,
                             xi_mc_se=xse*4 if xse is not None else None,
                             xi_over_side_95_bound=bound,
                             spectral_relative_95_bounds=spectral_bound.tolist(),
                             invalid_bootstrap_fraction=invalid,ancestry_alarms=alarms,
                             status='within_development_tolerances' if passes else 'unresolved',
                             ground_state_qualified=False))
        screen_paths = [root/'screen'/(name(s,gamma)+'.json') for s in ids]
        if all(p.exists() for p in screen_paths):
            values=[json.loads(p.read_text()) for p in screen_paths]
            for setting in ['base','equilibration','forward','combined']:
                raw=np.array([next(x['spectrum'] for x in v['settings'] if x['setting']==setting) for v in values])
                mean=raw.mean(0)
                screens.append(dict(gamma=gamma,setting=setting,xi=float(length(mean,2)),
                                    exact_xi=target_x*4,
                                    spectral_relative_bias=((mean-target)/target).tolist()))
    report=dict(rows=rows,finite_time=screens,incomplete=incomplete,
                qualification='Development comparisons only; held-out checks, stronger plateaus and larger sizes still required',
                uncertainty='Repeat bootstrap conditional on 16 fixed disorders; approximate pointwise 95% bounds; four repeats give limited tail resolution')
    (root/'convergence_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Correlation-length convergence review','',report['qualification'],'',report['uncertainty'],'',
           '| Γ | Setting | ξ GFMC | ξ exact | MC SE | Ancestry alarms | Assessment |',
           '|---:|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['gamma']} | {r['setting']} | {r['xi']} | {r['exact_xi']:.5f} | {r['xi_mc_se']} | {r['ancestry_alarms']} | {r['status']} |")
    lines.extend(['',f'Incomplete context/setting groups: {len(incomplete)}.',
                  'No failed runs or difficult disorders are removed to obtain agreement.'])
    (root/'CONVERGENCE_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete_groups=len(rows),incomplete_groups=len(incomplete),
                         within_tolerance=sum(r['status']=='within_development_tolerances' for r in rows))))


if __name__=='__main__':
    main()
