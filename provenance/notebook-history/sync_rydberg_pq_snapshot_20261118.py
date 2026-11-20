"""Retrieve a closed reduced-observable snapshot into local staging."""
from pathlib import Path
import base64
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

import nbformat
import numpy as np

root = Path('/Users/tianyiyan/ResearchLocal/spin-glasses/staging')
target = Path('/Users/tianyiyan/Dropbox/spin-glasses/notebook/rydberg_kagome_L4_Pq_progress.ipynb')
assert root.resolve() == root
assert shutil.disk_usage(root).free > 200_000_000
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if len(sys.argv) > 1:
    stage = Path(sys.argv[1])
    initial = json.loads((stage / 'sync-input.json').read_text())
    record = json.loads((stage / 'remote-export.json').read_text())
else:
    assert os.environ.get('SSHPASS'), 'AutoDL credentials must be available in the environment'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stage = root / f'rydberg-L4-notebook-sync-{stamp}'
    stage.mkdir(exist_ok=False)
    shutil.copy2(target, stage / 'before-sync.ipynb')
    initial = dict(target=str(target), original_sha256=sha(stage / 'before-sync.ipynb'),
                   started_utc=stamp, stage=str(stage))
    (stage / 'sync-input.json').write_text(json.dumps(initial, indent=2) + '\n')
    remote_script = '/root/autodl-tmp/spin-glasses/runs/rydberg-L4-production-20261117-v5/export_pq_preview.py'
    export = subprocess.run(
        ['sshpass', '-e', 'ssh', '-o', 'ConnectTimeout=15', 'autodl',
         f'OPENBLAS_NUM_THREADS=1 /root/miniconda3/bin/python -B {remote_script}'],
        check=True, capture_output=True, text=True, timeout=120)
    record = json.loads(export.stdout)
    (stage / 'remote-export.json').write_text(json.dumps(record, indent=2) + '\n')
    partial = stage / 'snapshot.tar.gz.part'
    subprocess.run(['sshpass', '-e', 'scp', '-o', 'ConnectTimeout=15',
                    'autodl:' + record['archive'], str(partial)], check=True, timeout=120)
    assert partial.stat().st_size == record['archive_bytes']
    assert sha(partial) == record['archive_sha256']
    partial.rename(stage / 'snapshot.tar.gz')
archive = stage / 'snapshot.tar.gz'
assert sha(archive) == record['archive_sha256']
with tarfile.open(archive) as tar:
    members = tar.getmembers()
    assert len(members) == 3
    assert {m.name for m in members} == {'histograms.npz', 'metadata.json', 'case-manifest.json'}
    assert all(m.isfile() for m in members)
    assert sum(m.size for m in members) < 100_000_000
    for member in members:
        (stage / member.name).write_bytes(tar.extractfile(member).read())
meta = json.loads((stage / 'metadata.json').read_text())
assert meta == record['metadata']
assert sha(stage / 'histograms.npz') == meta['files']['histograms.npz']
manifest = json.loads((stage / 'case-manifest.json').read_text())
assert len(manifest) == len({r['case'] for r in manifest}) == meta['case_count']
n = len(meta['sample_ids'])
assert meta['case_count'] == n * 180
geometry_hashes = {}
for row in manifest:
    sample = int(row['case'].split('_', 1)[0][1:])
    geometry_hashes.setdefault(sample, set()).add(row['geometry_sha256'])
assert set(geometry_hashes) == set(meta['sample_ids'])
assert all(len(hashes) == 1 for hashes in geometry_hashes.values())
with np.load(stage / 'histograms.npz', allow_pickle=False) as data:
    assert data['counts'].shape == (n, 3, 2, 30, 49)
    assert data['qea'].shape == data['passed'].shape == (n, 3, 2, 30)
    assert np.all(data['counts'] >= 0)
    assert meta['pairs_per_case'] == 20000
    assert np.all(data['counts'].sum(axis=-1) == meta['pairs_per_case'])
    assert np.isfinite(data['qea']).all() and np.isfinite(data['q2']).all()
    assert np.all((data['qea'] >= -1e-12) & (data['qea'] <= 1+1e-12))
    assert data['sample_ids'].tolist() == meta['sample_ids']
    assert (~data['passed']).sum() == len(meta['flags'])
    np.testing.assert_allclose(data['drives'], np.geomspace(.03, 10., 30))
    np.testing.assert_array_equal(data['models'], ['full', 'nn_nnn', 'nn'])
    np.testing.assert_array_equal(data['deltas'], [2., 3.15])
    previous = nbformat.read(stage / 'before-sync.ipynb', as_version=4)
    payload_line = next(line for line in previous.cells[1].source.splitlines() if line.startswith('PAYLOAD = '))
    import ast
    previous_bytes = base64.b64decode(ast.literal_eval(payload_line.split(' = ', 1)[1]))
    with np.load(io.BytesIO(previous_bytes), allow_pickle=False) as old:
        previous_n = len(old['sample_ids'])
        subset = [meta['sample_ids'].index(int(sid)) for sid in old['sample_ids']]
        for key in ['counts', 'qea', 'q2', 'passed']:
            np.testing.assert_array_equal(data[key][subset], old[key])

initial.update(remote=record['directory'], archive_sha256=sha(archive),
               payload_sha256=meta['files']['histograms.npz'], archive_bytes=archive.stat().st_size,
               verified=True, previous_geometries=previous_n, geometries=n,
               cases=meta['case_count'], snapshot_utc=meta['snapshot_utc'])
(stage / 'transfer-record.json').write_text(json.dumps(initial, indent=2) + '\n')
print(json.dumps(dict(stage=str(stage), previous_geometries=previous_n, geometries=n,
                     cases=meta['case_count'], snapshot_utc=meta['snapshot_utc'],
                     flags=meta['flags'], archive_sha256=sha(archive),
                     archive_bytes=archive.stat().st_size, previous_data_unchanged=True), indent=2))
