"""Refresh embedded data and snapshot labels, preserving current analysis code."""
from pathlib import Path
import ast
import base64
import datetime
import hashlib
import json
import os
import re
import sys

import nbformat
from nbclient import NotebookClient

stage = Path(sys.argv[1])
transfer = json.loads((stage / 'transfer-record.json').read_text())
assert transfer['verified']
meta = json.loads((stage / 'metadata.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(stage / 'histograms.npz') == meta['files']['histograms.npz']
assert sha(stage / 'before-sync.ipynb') == transfer['original_sha256']
nb = nbformat.read(stage / 'before-sync.ipynb', as_version=4)
sources = [cell.source for cell in nb.cells]
old_line = next(line for line in nb.cells[1].source.splitlines() if line.startswith('METADATA = '))
old_meta = json.loads(ast.literal_eval(ast.parse(old_line).body[0].value.args[0]))
old_n, n = len(old_meta['sample_ids']), len(meta['sample_ids'])
# The retained diagnostic discussion is valid only for the same flagged inputs.
assert old_meta['flags'] == meta['flags'], 'Update flag descriptions before publishing changed diagnostics'
assert meta['sample_ids'] == list(range(n))
snapshot_time = datetime.datetime.strptime(meta['snapshot_utc'], '%Y%m%dT%H%M%SZ')
timestamp_label = snapshot_time.strftime('%d %B %Y, %H:%M UTC').lstrip('0')
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace(f'{old_n} geometries', f'{n} geometries')
        cell.source = cell.source.replace(f'{old_n} matched geometries', f'{n} matched geometries')
        cell.source = cell.source.replace(f'S={old_n}', f'S={n}')
nb.cells[0].source, count = re.subn(r'\*\*Snapshot: .*?\*\*', f'**Snapshot: {timestamp_label}.**', nb.cells[0].source, count=1)
assert count == 1
nb.cells[0].source = nb.cells[0].source.replace(
    f'{old_n} completed geometries (IDs 0–{old_n-1})',
    f'{n} completed geometries (IDs 0–{n-1})').replace(
    f"{old_meta['case_count']:,} cases", f"{meta['case_count']:,} cases")
code = nb.cells[1].source
prefix, suffix = code[:code.index('METADATA =')], code[code.index('raw ='):]
nb.cells[1].source = (prefix + 'METADATA = json.loads(' + repr(json.dumps(meta)) + ')\n'
                      + 'PAYLOAD = ' + repr(base64.b64encode((stage / 'histograms.npz').read_bytes()).decode())
                      + '\n' + suffix)
nb.cells[10].source, count = re.subn(
    r'(Snapshot archive SHA-256: `)[0-9a-f]{64}(`)',
    lambda m: m.group(1) + transfer['archive_sha256'] + m.group(2), nb.cells[10].source)
assert count == 1
nb.cells[10].source, count = re.subn(
    r'(Archive, manifest and transfer record: `)[^`]+(`)',
    lambda m: m.group(1) + str(stage).replace(str(Path.home()), '~') + '/' + m.group(2),
    nb.cells[10].source)
assert count == 1
nb.metadata['snapshot_utc'] = meta['snapshot_utc']
if n == 128:
    assert meta['case_count'] == 23040
    nb.cells[0].source = nb.cells[0].source.replace(
        '# Rydberg Kagome L4: preliminary P(q) distributions',
        '# Rydberg Kagome L4: full 128-geometry P(q) dataset').replace(
        'The intended ensemble has 128 geometries; these are preliminary results at **chi_RG=12, D_MPS=128**, not RG-converged predictions.',
        'The planned production ensemble is complete: **128/128 geometries and 23,040/23,040 cases** at **chi_RG=12, D_MPS=128**. Six solver flags remain unresolved; these are not RG-converged predictions.')
    nb.cells[10].source = nb.cells[10].source.replace(
        'Use the common completed cohort for these preliminary comparisons. The final target remains all 128 geometries.',
        'All comparisons now use the complete matched ensemble of 128 geometries. Production completion does not resolve the six solver flags or the measured RG sensitivity.')
    nb.metadata['production_ensemble_complete'] = True
for i in [2,3,5,7,9]:
    assert nb.cells[i].source == sources[i]
for cell in nb.cells:
    assert not any(ord(ch) < 32 and ch not in '\n\t' for ch in cell.source)
    if cell.cell_type == 'markdown':
        assert f'{old_n} geometries' not in cell.source
        assert f'{old_n} matched geometries' not in cell.source

os.environ['MPLCONFIGDIR'] = str(stage / 'mpl-cache')
NotebookClient(nb, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(stage)}}).execute()
nbformat.validate(nb)
figure_count = 0
for i, cell in enumerate(nb.cells):
    for output in cell.get('outputs', []):
        assert output.output_type != 'error'
        if 'image/png' in output.get('data', {}):
            figure_count += 1
            (stage / f'plot-{figure_count}.png').write_bytes(base64.b64decode(output.data['image/png']))
        if output.output_type == 'stream':
            print(output.text)
        if i == 9 and 'text/markdown' in output.get('data', {}):
            print(output.data['text/markdown'])
assert figure_count == 4
target = Path(transfer['target'])
candidate = stage / target.name
nbformat.write(nb, candidate)
readback = nbformat.read(candidate, as_version=4)
nbformat.validate(readback)
assert readback.metadata['snapshot_utc'] == meta['snapshot_utc']
publication = dict(target=str(target), candidate=str(candidate),
    original_sha256=transfer['original_sha256'], candidate_sha256=sha(candidate),
    snapshot_utc=meta['snapshot_utc'], geometries=n, cases=meta['case_count'],
    flag_count=len(meta['flags']), figures=figure_count, executed=True,
    analysis_code_preserved=True)
(stage / 'publication-ready.json').write_text(json.dumps(publication, indent=2) + '\n')
print(json.dumps(publication, indent=2))
