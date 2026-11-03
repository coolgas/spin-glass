"""One-shot verified reference/anchor diagnostic snapshot for the notebook."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .collective_campaign import write
from .correlation_lengths import sha


def collect(root):
    status_path = root/'results'/('parallel_status.json' if root.name.startswith('l4') else 'status.json')
    status = json.loads(status_path.read_text())
    runs = []
    for path in sorted((root/'results').glob('**/runs/*.json')):
        record = json.loads(path.read_text())
        raw = path.with_suffix('.npz')
        assert sha(raw) == record['raw_sha256'], raw
        mass = np.asarray(record['mass'])
        grid = np.asarray(record['q_grid'])
        assert np.isfinite(mass).all() and (mass >= 0).all()
        np.testing.assert_allclose(mass.sum(), 1., atol=1e-12)
        np.testing.assert_allclose([grid@mass, grid**2@mass], [record['q'], record['q2']], atol=1e-12)
        with np.load(raw) as data:
            ess = data['descendant_ess']; lineage = data['largest_lineage']
            assert np.isfinite(ess).all() and np.isfinite(lineage).all()
            stats = dict(ess_quantiles=np.quantile(ess, [0, .1, .5, .9, 1]).tolist(),
                lineage_quantiles=np.quantile(lineage, [0, .1, .5, .9, 1]).tolist(),
                ess_alarm_fraction=float(np.mean(ess < 8)), lineage_alarm_fraction=float(np.mean(lineage > .5)))
        assert bool(ess.min() < 8 or lineage.max() > .5) == record['ancestry_alarm']
        runs.append(dict(path=str(path), json_sha256=sha(path), record=record, **stats))
    return dict(root=str(root), status=status, runs=runs,
                raw_verified=len(runs), ancestry_alarms=sum(r['record']['ancestry_alarm'] for r in runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    names = ['l6-convergence-20261103-v1', 'extended-rbm-20261103-v1',
             'ancestry-diagnostic-20261103-v1', 'l6-reference-fields-20261103-v1']
    campaigns = [collect(args.base/name) for name in names]
    record = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), campaigns=campaigns,
                  ground_state_qualified=False)
    args.destination.mkdir(parents=True, exist_ok=True)
    write(args.destination/'reference_progress.json', record)
    lines = ['# L6 lower-field and reference comparison', '',
        f"Verified snapshot: {record['snapshot_utc']}. No campaign is qualified by this report.", '',
        '| Campaign | Training milestones | GFMC runs | Ancestry alarms | Task failures |',
        '|---|---:|---:|---:|---:|']
    for campaign in campaigns:
        status = campaign['status']
        lines.append(f"| {Path(campaign['root']).name} | {status.get('completed_milestones', status.get('completed_extensions', '—'))} | {len(campaign['runs'])} | {campaign['ancestry_alarms']} | {len(status['failures'])} |")
    lines.extend(['', 'Reference scope: Gamma .85/1.00; disorders60000/60001; both RBM seeds;',
        '400/1600/4000 target updates; at most 48 GFMC runs. All other jobs continue.',
        'Compare with lower-field .55/.60/.65/.70 at fixed h=.15. Existing Gamma4 controls',
        'remain in the notebook. Reference fields are not assumed to be outside a glassy regime.', '',
        '| Gamma | Disorder | Seed | Training budget path | Repeat | Energy/site | q2 | ESS min / median | Max lineage | Alarm |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---|'])
    for campaign in campaigns:
        for run in campaign['runs']:
            r = run['record']; t = r['task']; ess = run['ess_quantiles']
            budget = '/'.join(Path(run['path']).parts[-5:-2])
            lines.append(f"| {t['gamma']:.2f} | {t['sample_id']} | {t['training_seed']} | {budget} | {t['repeat']} | {r['energy']/108:.6f} | {r['q2']:.5f} | {ess[0]:.2f} / {ess[2]:.2f} | {r['largest_lineage']:.3f} | {r['ancestry_alarm']} |")
    lines.extend(['', 'Raw SHA256, histogram normalization/moments, finite ancestry and alarm consistency',
        'were checked for every listed run. JSON retains full histograms, connected spectra,',
        'energy, provenance and ancestry quantiles/fractions; raw arrays remain in their campaign paths.',
        'Alarms and seed/repeat differences prevent a physical interpretation until resolved.',
        'Two repeats and two fixed disorders give limited precision. No automatic promotion,',
        'phase label or transition estimate follows from higher-field improvement.'])
    (args.destination/'REFERENCE_FIELDS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({c['root']: dict(runs=len(c['runs']), alarms=c['ancestry_alarms']) for c in campaigns}, indent=2))


if __name__ == '__main__':
    main()
