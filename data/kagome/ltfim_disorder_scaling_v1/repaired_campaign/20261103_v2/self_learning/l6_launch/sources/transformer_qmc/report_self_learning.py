"""Validate completed L2 feedback artifacts and emit a compact research report."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .physics import basis, exact_reference, make_af_sample
from .self_learning_rbm import exact_evaluation
from .simple_guides import RBM


def run(pilot, output):
    ham = make_af_sample(2, 60000).hamiltonian(.15)
    truth = exact_reference(ham, .6); states = basis(12)
    rows = []; hashes = {}
    for seed in [0, 1]:
        root = pilot/f'seed{seed}'
        manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['settings']['stints'] == 6
        assert manifest['settings']['L'] == 2 and manifest['settings']['gamma'] == .6
        assert manifest['settings']['sample_id'] == 60000
        summary = json.loads((root/'summary.json').read_text())['rows']
        assert [r['stint'] for r in summary] == list(range(6))
        assert json.loads((root/'status.json').read_text())['state'] == 'completed_unqualified'
        previous = None
        for row in summary:
            folder = root/f"stint_{row['stint']:03d}"
            for name, key in [('model.pt', 'model_sha256'), ('input_model.pt', 'input_model_sha256'),
                              ('walkers.npz', 'raw_sha256'), ('fit.npz', 'fit_sha256')]:
                path = folder/name; digest = sha(path)
                assert digest == row[key], str(path)
                hashes[str(path.relative_to(pilot))] = digest
            with np.load(folder/'walkers.npz') as raw:
                samples = raw['samples'].copy(); lineage = raw['lineages'].copy()
                assert samples.shape == (7680, 12)
                assert np.isin(samples, [-1, 1]).all()
                assert raw['energies'].shape == (200, 2)
                for key in raw.files:
                    assert np.isfinite(raw[key]).all(), key
            with np.load(folder/'fit.npz') as fit:
                assert fit['gradient_trace'][-1, 0] == 1000
                assert np.isfinite(fit['gradient_trace']).all()
            collection = lineage[80:]
            np.testing.assert_allclose(collection[:, 0].min(), row['minimum_training_lineage_ess'])
            np.testing.assert_allclose(collection[:, 1].max(), row['largest_training_lineage'])
            before = RBM(12, 2); after = RBM(12, 2)
            before.load_state_dict(torch.load(folder/'input_model.pt', map_location='cpu', weights_only=False)['model'])
            after.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=False)['model'])
            if previous is not None:
                for key, value in before.state_dict().items():
                    torch.testing.assert_close(value, previous[key], atol=0, rtol=0)
            previous = after.state_dict()
            for key, model in [('before', before), ('after', after)]:
                evaluation = exact_evaluation(model, ham, .6, samples)
                for metric, value in evaluation.items():
                    np.testing.assert_allclose(value, row[key][metric], atol=1e-10, rtol=1e-10)
            with torch.no_grad():
                logs = before(torch.tensor(states, dtype=torch.float64)).numpy()
            ideal = np.exp(logs-logs.max())*truth['amplitude']; ideal /= ideal.sum()
            ids = ((1-samples.astype(np.int64))//2)@(1 << np.arange(12))
            empirical = np.bincount(ids, minlength=4096)/len(ids)
            rows.append(dict(seed=seed, **row,
                empirical_vs_ideal_mixed_tv=float(np.abs(empirical-ideal).sum()/2)))
        checkpoint = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=False)
        assert checkpoint['phase'] == 'completed_unqualified' and checkpoint['stint'] == 6
        for key, value in previous.items():
            torch.testing.assert_close(value, checkpoint['model'][key], atol=0, rtol=0)
        for name in ['checkpoint.pt', 'manifest.json', 'status.json', 'summary.json']:
            hashes[str((root/name).relative_to(pilot))] = sha(root/name)
    report = dict(rows=rows, verified_artifacts=len(hashes),
        exact_energy=truth['energy'], exact_residual=truth['residual'],
        completed_stints=12, ground_state_qualified=False,
        interpretation='Feedback improves both guides; exact guide energy and density errors remain. Independent measurement validation is separate.')
    write(output/'training_report.json', report)
    write(output/'training_sha256.json', hashes)
    print(json.dumps(dict(verified_artifacts=len(hashes), completed_stints=12,
        finals=[dict(seed=s, **next(r for r in rows if r['seed']==s and r['stint']==5)['after']) for s in [0, 1]])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run(args.pilot, args.output)


if __name__ == '__main__':
    main()
