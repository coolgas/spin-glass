"""Verify frozen source, notebook payload and matched-cohort snapshot."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

repo = Path(__file__).resolve().parents[1]
data = Path(sys.argv[1]) if len(sys.argv) > 1 else repo.parent / 'data/snapshot-20261119T005408Z'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
release = json.loads((repo / 'provenance/production-release/manifest.json').read_text())
for name, expected in release['files'].items():
    assert sha(repo / name) == expected, name
for name, expected in release['artifacts'].items():
    assert sha(repo / 'configs' / name) == expected, name
notebook = json.loads((repo / 'notebook/rydberg_kagome_L4_Pq_progress.ipynb').read_text())
payload = None
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    for node in ast.parse(''.join(cell['source'])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'PAYLOAD':
            payload = base64.b64decode(ast.literal_eval(node.value))
meta = json.loads((data / 'metadata.json').read_text())
assert hashlib.sha256(payload).hexdigest() == sha(data / 'histograms.npz') == meta['files']['histograms.npz']
cases = json.loads((data / 'case-manifest.json').read_text())
assert len(cases) == len({r['case'] for r in cases}) == 23040
with np.load(data / 'histograms.npz', allow_pickle=False) as z:
    assert z['counts'].shape == (128, 3, 2, 30, 49)
    assert np.all(z['counts'] >= 0) and np.all(z['counts'].sum(axis=-1) == 20000)
    assert int((~z['passed']).sum()) == len(meta['flags']) == 6
    np.testing.assert_array_equal(z['sample_ids'], np.arange(128))
    np.testing.assert_array_equal(z['models'], ['full', 'nn_nnn', 'nn'])
    np.testing.assert_array_equal(z['deltas'], [2., 3.15])
    np.testing.assert_allclose(z['drives'], np.geomspace(.03, 10., 30))
    assert np.isfinite(z['qea']).all() and np.isfinite(z['q2']).all()
print('PASS: frozen source/config hashes, notebook payload, 128 disorders, 23040 cases, six retained flags')
