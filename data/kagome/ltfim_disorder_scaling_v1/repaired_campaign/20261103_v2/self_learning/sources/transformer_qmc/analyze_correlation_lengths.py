"""Validate and reduce the data used in the concise correlation-length notebook."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .correlation_lengths import BASE, FIELD, GAMMAS, IDS, OUT, ROOT, length, sha
from .extend_correlation_fields import (DESTINATION as EXTENSION, FIELDS,
                                       expected_manifest, load_inputs, run_path,
                                       validate_record)
from .physics import basis, make_af_sample
from .production_protocol import exact_spectra, geometry


def paired_summary(exact, runs, size=2, draws=4000):
    """Keep momenta, matched disorders, and independent repeats paired."""
    exact, runs = np.asarray(exact), np.asarray(runs)
    if runs.ndim != 3 or runs.shape[0] != len(exact) or runs.shape[-1] != 4:
        raise ValueError('Expected disorder × repeat × four spectra')
    rng = np.random.default_rng(20261102)
    D, R, _ = runs.shape
    disorder = rng.integers(D, size=(draws, D))
    repeats = rng.integers(R, size=(draws, D, R))
    boot_exact = length(exact[disorder].mean(axis=1), size)
    boot_gfmc = length(runs[disorder[..., None], repeats].mean(axis=(1, 2)), size)
    e, g = float(length(exact.mean(axis=0), size)), float(length(runs.mean(axis=(0, 1)), size))
    def se(values):
        finite = values[np.isfinite(values)]
        return float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
    return dict(exact=e, exact_se=se(boot_exact), gfmc=g, gfmc_se=se(boot_gfmc),
                delta=g-e, delta_se=se(boot_gfmc-boot_exact),
                invalid_bootstrap_fraction=float(np.mean(~np.isfinite(boot_gfmc))),
                disorders=D, repeats=R)


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes, exact, raw_rows = {}, {}, []
    manifest_path = OUT / 'l2_gfmc/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest['source_sha256'].items():
        assert sha(ROOT / relative) == digest
    hashes[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
    run_sources = {gamma: (manifest_path.parent, manifest) for gamma in GAMMAS}
    extension_counts = {}
    extension_manifest_path = EXTENSION / 'manifest.json'
    if extension_manifest_path.exists():
        extension_manifest = json.loads(extension_manifest_path.read_text())
        assert extension_manifest == expected_manifest(), 'Changed field-extension identity'
        hashes[str(extension_manifest_path.relative_to(ROOT))] = sha(extension_manifest_path)
        expected_paths = {run_path(gamma, sid, repeat)
                          for gamma in FIELDS for sid in IDS for repeat in range(4)}
        assert set(EXTENSION.glob('*_repeat*.json')) <= expected_paths
        for gamma in FIELDS:
            count = 0
            for sid in IDS:
                _, _, inputs, checkpoint = load_inputs(gamma, sid)
                for repeat in range(4):
                    path = run_path(gamma, sid, repeat)
                    if path.exists():
                        validate_record(json.loads(path.read_text()), gamma, sid, repeat,
                                        inputs, checkpoint)
                        hashes[str(path.relative_to(ROOT))] = sha(path)
                        count += 1
            extension_counts[str(gamma)] = count
            if count == len(IDS) * 4:
                run_sources[gamma] = (EXTENSION, extension_manifest)
    finite_time_path = OUT / 'l2_finite_time_check.json'
    if finite_time_path.exists():
        hashes[str(finite_time_path.relative_to(ROOT))] = sha(finite_time_path)
    signature = json.loads((BASE / 'pipeline/manifest.json').read_text())['checks_signature']
    spins, lattice = basis(12).astype(float), geometry(2)
    all_gammas = [.25, .5, .75, 1., 1.5, 2., 3.]
    for gamma in all_gammas:
        for sid in IDS:
            stem = f'L2_s{sid}_g{gamma:g}_h{FIELD:g}_K1'.replace('.', 'p')
            path = BASE / 'qualification/exact' / (stem + '.json')
            meta = json.loads(path.read_text())
            assert meta['status'] == 'completed' and meta['signature'] == signature
            assert meta['residual'] <= 1e-9
            assert (meta['L'], meta['sample_id'], meta['gamma'], meta['h']) == (2, sid, gamma, FIELD)
            assert sha(path.with_suffix('.npz')) == meta['npz_sha256']
            for source in (path, path.with_suffix('.npz')):
                hashes[str(source.relative_to(ROOT))] = sha(source)
            sample = make_af_sample(2, sid)
            with np.load(path.with_suffix('.npz')) as archive:
                np.testing.assert_array_equal(archive['bonds'], sample.bonds)
                np.testing.assert_array_equal(archive['J'], sample.J)
                psi = archive['amplitude']
                np.testing.assert_allclose(psi @ psi, 1., atol=1e-10, rtol=0)
                with threadpool_limits(limits=1):
                    spectrum = exact_spectra(psi**2, spins, lattice)['connected_spectrum']
            np.testing.assert_allclose(spectrum, meta['observables']['connected_spectrum'], atol=2e-10)
            raw = spectrum[[0, *lattice['kmin_indices']]]
            exact[gamma, sid] = raw
            raw_rows.append(dict(L=2, gamma=gamma, h=FIELD, sample_id=sid, method='exact',
                                 repeat=-1, **dict(zip(['S0', 'Sk1', 'Sk2', 'Sk3'], raw))))
    summaries, gfmc_records, disorder_rows = [], [], []
    for gamma in all_gammas:
        e = np.array([exact[gamma, sid] for sid in IDS])
        if gamma in run_sources:
            run_directory, run_manifest = run_sources[gamma]
            runs = []
            for sid in IDS:
                repeat_raw = []
                for repeat in range(4):
                    stem = f'L2_s{sid}_g{gamma:g}_h{FIELD:g}_K1'.replace('.', 'p')
                    path = run_directory / (stem + f'_repeat{repeat}.json')
                    r = json.loads(path.read_text())
                    assert (r['L'], r['gamma'], r['h'], r['sample_id'], r['repeat']) == (2, gamma, FIELD, sid, repeat)
                    assert r['parameters'] == run_manifest['parameters'] == manifest['parameters']
                    assert np.isfinite(r['result']['connected_spectrum']).all()
                    for relative, digest in r['input_sha256'].items():
                        assert sha(ROOT / relative) == digest
                    hashes[str(path.relative_to(ROOT))] = sha(path)
                    gfmc_records.append(r)
                    s = np.asarray(r['result']['connected_spectrum'])[[0, 1, 2, 3]]
                    np.testing.assert_allclose(s[0], r['result']['chi_connected'], atol=1e-12)
                    repeat_raw.append(s)
                    raw_rows.append(dict(L=2, gamma=gamma, h=FIELD, sample_id=sid,
                                         method='guided-GFMC', repeat=repeat,
                                         **dict(zip(['S0', 'Sk1', 'Sk2', 'Sk3'], s))))
                runs.append(repeat_raw)
                disorder_rows.append(dict(
                    L=2, gamma=gamma, h=FIELD, sample_id=sid,
                    exact_xi=float(length(exact[gamma, sid], 2)),
                    gfmc_xi=float(length(np.mean(repeat_raw, axis=0), 2))))
            row = paired_summary(e, runs)
            matched = [r for r in gfmc_records if r['gamma'] == gamma]
            row['ancestry_alarms'] = sum(r['ancestry_alarm'] for r in matched)
            row['comparison_status'] = ('unresolved ancestry' if row['ancestry_alarms'] else
                                        'discrepancy' if abs(row['delta']) > 2 * row['delta_se'] else
                                        'compatible within uncertainty')
        else:
            # Reuse the same resampling of disorders for the exact-only curve.
            row = paired_summary(e, e[:, None, :])
            for key in ('gfmc', 'gfmc_se', 'delta', 'delta_se', 'invalid_bootstrap_fraction'):
                row[key] = np.nan
            row['repeats'], row['ancestry_alarms'] = 0, 0
            row['comparison_status'] = 'exact only'
        summaries.append(dict(L=2, gamma=gamma, h=FIELD, **row))
    table = pd.DataFrame(summaries)
    table.to_csv(OUT / 'l2_comparison.csv', index=False)
    pd.DataFrame(raw_rows).to_csv(OUT / 'l2_raw_spectra.csv', index=False)
    pd.DataFrame(disorder_rows).to_csv(OUT / 'l2_by_disorder.csv', index=False)

    # Use one complete size series with identical numerical settings throughout.
    pipeline = BASE / 'pipeline'
    task_path = pipeline / 'tasks.json'
    hashes[str(task_path.relative_to(ROOT))] = sha(task_path)
    tasks = json.loads(task_path.read_text())
    size_rows, size_raw, settings = [], [], None
    for L in range(4, 11):
        selected = [t for t in tasks if t['stage'] == 'direct_feasibility' and
                    t['method'] == 'network' and t['L'] == L and
                    t['gamma'] == .5 and t['h'] == FIELD and t['precision'] == 'float32' and
                    (pipeline / 'results' / (t['id'] + '.json')).exists()]
        assert len(selected) == 1 and {t['repeat'] for t in selected} == {0}
        assert {t['sample_id'] for t in selected} == {60000}
        runs = []
        for task in sorted(selected, key=lambda t: t['repeat']):
            if settings is None:
                settings = task['parameters']
            assert task['parameters'] == settings
            assert task['precision'] == 'float32'
            path = pipeline / 'results' / (task['id'] + '.json')
            r = json.loads(path.read_text())
            assert r['state'] == 'completed' and r['task'] == task
            assert not r['ground_state_qualified']
            assert r['checkpoint_sha256'] == gfmc_records[0]['checkpoint_sha256']
            hashes[str(path.relative_to(ROOT))] = sha(path)
            s = np.asarray(r['result']['connected_spectrum'])[[0, 1, L, L+1]]
            runs.append(s)
            size_raw.append(dict(L=L, sample_id=60000, repeat=task['repeat'],
                                 **dict(zip(['S0', 'Sk1', 'Sk2', 'Sk3'], s))))
        runs = np.asarray(runs)
        size_rows.append(dict(L=L, gamma=.5, h=FIELD, xi=float(length(runs.mean(0), L)),
                              repeat_se=np.nan, invalid_bootstrap_fraction=np.nan,
                              disorders=1, repeats=1, ground_state_qualified=False))
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(OUT / 'size_comparison.csv', index=False)
    pd.DataFrame(size_raw).to_csv(OUT / 'size_raw_spectra.csv', index=False)
    analysis_sources = [Path(__file__), ROOT / 'code/transformer_qmc/report_correlation_lengths.py']
    provenance = dict(input_sha256=hashes,
                      analysis_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in analysis_sources},
                      L2_parameters=gfmc_records[0]['parameters'],
                      size_parameters=settings, size_precision='float32',
                      L2_run_count=len(gfmc_records), exact_reference_count=len(exact),
                      L2_gfmc_fields=sorted(run_sources), extension_runs_by_field=extension_counts,
                      L2_ancestry_alarms=sum(r['ancestry_alarm'] for r in gfmc_records),
                      bootstrap_draws=4000, bootstrap_seed=20261102,
                      length_units='nearest-neighbor spacing; physical side=2L',
                      size_ground_state_qualified=False)
    (OUT / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return table, sizes, provenance
