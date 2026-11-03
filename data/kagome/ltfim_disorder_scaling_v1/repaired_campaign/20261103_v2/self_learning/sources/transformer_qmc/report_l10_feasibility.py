"""Summarize bounded L10 cost probes without claiming ground-state accuracy."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/kagome'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/kagome/transformer_af_l10_feasibility_v1/ASSESSMENT.md'))
    args = parser.parse_args()
    records = []
    for version in (1, 2, 3):
        folder = args.root / f'transformer_af_l10_feasibility_v{version}'
        path = folder / 'results/benchmark.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for name, expected in data['source_sha256'].items():
            assert hashlib.sha256((folder/'code'/name).read_bytes()).hexdigest() == expected, name
        records.append((version, data))
    if not records:
        raise FileNotFoundError('No mirrored L10 benchmarks')
    lines = [
        '# L=10 (300-spin PBC kagome) measured feasibility',
        '',
        '## Scope and status',
        '',
        'Bounded AutoDL H800 probes using the frozen seed501 checkpoint, one disorder '
        '(ID 10000), Gamma=0.5, and h=0 or 0.3. These are execution and cost measurements, '
        'not converged ground-state or phase-boundary results. Probe source snapshots '
        'are retained and their recorded hashes verified by this report generator.',
        '',
    ]
    for version, data in records:
        lines.append(f'- v{version}: **{data["status"]}**, {data["elapsed_seconds"]:.1f} s; '
                     f'{data["started_utc"]}; precision: {data["dtype"]}.')
    lines += ['', '## Current float64 inference', '',
              '| L | Spins | h | Configurations | All-flip seconds | Peak allocated GiB |',
              '|---:|---:|---:|---:|---:|---:|']
    for _, data in records:
        for row in data['rows']:
            if row['stage'] == 'all_flip' and row['configurations'] == 128:
                lines.append(f'| {row["L"]} | {3*row["L"]**2} | {row["h"]} | 128 | '
                             f'{row["seconds"]:.3f} | {row["peak_allocated_mib"]/1024:.3f} |')
    lines += ['', 'All-flip queries evaluate the ratios needed to choose a single-spin event. '
              'The current projector repeatedly requests these ratios during propagation; '
              'its exact state cache avoids repeated queries when states recur. '
              'Peak allocated memory is PyTorch allocation, not total device memory or a production memory estimate.',
              'The v1 128-configuration float64 L=10 batches allocate about 1.93 GiB. '
              'The longer v2 probe also completed with 128 four-replica groups (512 configurations), '
              'using about 2.18 GiB peak allocated memory. Production populations and cache storage '
              'need their own memory measurements.',
              '', '## End-to-end short projection', '',
              '| Probe | L | h | Inference | Four-replica walker groups | Total imaginary time | Wall seconds | Inference seconds | Minimum descendant ESS |',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    projections = []
    for version, data in records:
        for row in data['rows']:
            if row['stage'] not in ('project', 'precision_project'):
                continue
            precision = 'float32' if row['stage'] == 'precision_project' else 'float64'
            projections.append((version, precision, row))
            lines.append(f'| v{version} | {row["L"]} | {row["h"]} | {precision} | '
                         f'{row["settings"]["n_walkers"]} | {row["total_projection_time"]:.3f} | '
                         f'{row["seconds"]:.3f} | {row["query_seconds"]:.3f} | '
                         f'{min(row["result"]["descendant_ess"]):.2f} |')
    lines += ['', 'Healthy ancestry over these very short forward times does not predict ancestry at '
              'production forward times. Each projector context has one seed; no disorder average '
              'or confidence interval on accuracy is inferred.', '',
              '## Float32 speed/error probe (candidate only)', '',
              'The precision probe uses full float32 matmul (highest setting), 128 guide-drawn and '
              '32 independent random configurations. Errors compare the same configurations against '
              'float64. The projector adapter retains double-precision NumPy rate/weight arithmetic. '
              'Changing inference precision is not yet qualified for production.', '',
              '| h | Chunk | Float64 seconds | Float32 seconds | Speed ratio | Max relative rate error | Max local-energy/site error |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for _, data in records:
        for row in data['rows']:
            if row['stage'] != 'precision_probe':
                continue
            lines.append(f'| {row["h"]} | {row["chunk"]} | {row["float64_seconds"]:.3f} | '
                         f'{row["seconds"]:.3f} | {row["float64_seconds"]/row["seconds"]:.1f}x | '
                         f'{row["max_abs_relative_rate_error"]:.3g} | '
                         f'{row["max_abs_local_energy_per_site_error"]:.3g} |')
    lines += ['', 'These finite sample discrepancies do not bound errors for other disorders, fields, '
              'states or long projections. Precision changes require exact-reference checks at L=2 '
              'and independent-ensemble comparisons of energy, overlap moments and connected correlations '
              'at larger sizes. A matching random seed need not produce matching trajectories.', '',
              '## Cost stress test, not a production forecast', '',
              'For scale only, extrapolate the W=32, total-time=0.2 probe to W=2048 and '
              'total time T=32+8+16*8=168 (the old 12-spin main settings). These settings are '
              'NOT qualified at L=10. Assume propagation cost linear in walker count and time, '
              'subtract measured sampling setup, and add setup scaled with walker count:', '',
              '`estimated_seconds = (seconds - sample_seconds)*(2048/32)*(168/0.2) + sample_seconds*(2048/32)`', '',
              '| Inference | h | Illustrative days per single population run |',
              '|---|---:|---:|']
    for _, precision, row in projections:
        if row['L'] == 10 and row['settings']['n_walkers'] == 32 and abs(row['total_projection_time']-.2) < 1e-10:
            estimate = ((row['seconds']-row['sample_seconds'])*64*840 + row['sample_seconds']*64)/86400
            lines.append(f'| {precision} | {row["h"]} | {estimate:.2f} |')
    lines += ['', 'Cache hit rates, state distributions, event rates, long-time ancestry, batch efficiency '
              'and concurrency can all change these costs; no billed GPU-hour prediction is established. '
              'The purpose is to show why successful execution alone does not make the planned '
              'multi-disorder campaign affordable.', '',
              '## Decision and next gate', '',
              '- The existing architecture/checkpoint can execute 300-spin PBC inference and short GFMC.',
              '- The current float64 path is not a practical basis for the full proposed campaign '
              'at the example W=2048/time=168 settings; inference dominates the measured runtime.',
              '- The isolated float32 candidate gives about 20x faster short propagation on both '
              'tested fields. It is a promising optimization, not a production accuracy certificate; '
              'the conditional large-run cost remains substantial.',
              '- Production-scope decision, 27 October 2026: include L=10 under the same full '
              '49-context survey and computation/acceptance protocol as L=4,6,8. Initially use '
              '64 fresh disorders, four repeats and the same 16-disorder unguided comparison subset; '
              'qualification and stronger audits are additional.',
              '- Use the same observables, precision targets, convergence ladders and release checks '
              'at every size. Converged populations and times may differ by size/context; '
              'successful short execution does not establish qualification.',
              '- Optimize measured inference cost and validate numerical precision before scaling populations.',
              '- Benchmark larger populations and longer forward times on multiple disorders and fields; '
              'then price the common production design using converged correlation observables. '
              'Schedule matched crossing cuts first while retaining full anchor-grid coverage '
              'as required work; budget-limited unfinished contexts remain pending.',
              '- Report a failure to qualify or an unaffordable configuration explicitly rather than '
              'substituting many short biased runs.', '',
              'Source: `results/benchmark.json` and `results/gpu.csv` under the v1/v2/v3 feasibility directories. '
              'Run `PYTHONPATH=code python -m transformer_qmc.report_l10_feasibility` to regenerate.', '',
              'Precision reference: PyTorch, Numerical accuracy (official documentation, '
              '`https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html`).', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
