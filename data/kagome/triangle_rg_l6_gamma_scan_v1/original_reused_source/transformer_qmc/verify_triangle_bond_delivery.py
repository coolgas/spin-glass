"""Verify a downloaded all-bond scan and install its self-contained notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import nbformat
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True)
    args = parser.parse_args()
    root = args.delivery/'results'
    status = json.loads((root/'status.json').read_text())
    assert status['state'] == 'complete' and status['completed'] == 256, status
    count = 0
    for folder,expected in [('cases',192),('exact',64)]:
        records = list((root/folder).glob('*.json'))
        assert len(records) == expected
        original_by_size = None
        for path in records:
            row = json.loads(path.read_text())
            raw = path.with_suffix('.npz')
            assert digest(raw) == row['raw_sha256'], raw
            with np.load(raw) as data:
                assert np.array_equal(data['J'],row['scale']*data['original_J'])
                if original_by_size is None:
                    original_by_size = data['original_J'].copy()
                assert np.array_equal(data['original_J'],original_by_size)
                if folder == 'cases':
                    counts = np.bincount((data['spins'][::2]==data['spins'][1::2]).sum(1),minlength=49)
                    assert np.array_equal(counts,data['counts'])
                    assert np.array_equal(counts/counts.sum(),row['mass'])
                    assert row['intertriangle_multiplier'] == 1.
                else:
                    assert abs(np.array(row['mass']).sum()-1.) < 1e-10
            count += 1
    provenance = json.loads((root/'report_provenance.json').read_text())
    assert digest(args.delivery/'report_sources'/'triangle_effective_bond_report.py') == provenance['report_source_sha256']
    assert digest(args.delivery/'report_sources'/'triangle_effective_bond_scan.py') == provenance['calculation_source_sha256']
    notebook = root/'triangle_effective_bond_scan.ipynb'
    assert digest(notebook) == provenance['notebook_sha256']
    nb = nbformat.read(notebook,as_version=4)
    nbformat.validate(nb)
    shutil.copy2(notebook,args.notebook)
    assessment = json.loads((root/'assessment.json').read_text())
    result = dict(raw_hashes_verified=count,l4_histograms_reconstructed=192,
        common_base_disorder_verified=True,all_bonds_scaled=True,notebook_validated=True,
        notebook_sha256=digest(notebook),source_snapshot_sha256=digest(args.delivery/'source_snapshot.tar.gz'),
        report_provenance_verified=True,
        candidates=sum(r['candidate_only'] for r in assessment['panels']),
        resolved_q2_candidate_pairs=sum(r['resolved_q2_candidate_only'] for r in assessment['shifts']))
    (root/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
