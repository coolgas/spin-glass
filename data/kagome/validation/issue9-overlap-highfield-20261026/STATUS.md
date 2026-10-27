# Gamma/sigma=5 overlap production check

Completed successfully on AutoDL at 2026-10-26 02:59:20 UTC in 1275.93 seconds. All 32 independent populations finished. All 73 remote artifacts (635,997 bytes) were downloaded and SHA256-verified; `verify_archives.py` independently confirmed statistics, settings, achieved times and lineage diagnostics. The instance remains powered on. Full settings and source hashes are in `campaign.json`; the observable convention and analysis rules are in `PROTOCOL.md`.

Local and remote smoke checks reproduced all 67 result fields exactly. The remote 128-walker check took approximately 0.59 seconds per run and recorded shortened forward cycles. That small-population result is a diagnostic check, not a conclusion about the 10,000-walker campaign.

All 256 forward cycles attained the requested time; none were truncated.

| Forward time | chi SG mean ± SE | finite-size q EA | Minimum descendant Kish size |
|---:|---:|---:|---:|
| 0.1 | 1.046178 ± 0.006875 | 0.098422 | 887.23 |
| 0.2 | 1.081062 ± 0.024506 | 0.100049 | 62.36 |
| 0.4 | 1.052394 ± 0.013094 | 0.098714 | 55.68 |
| 0.8 | 1.020229 ± 0.023033 | 0.097193 | 20.84 |

The 0.8-minus-0.4 susceptibility change is −0.032165 with Welch 95% interval [−0.090418, 0.026087]. None of the three adjacent-duration contrasts resolves a change. This supports production at the measured precision; it does not establish a stricter unprovided accuracy target. Use the longest tested forward time **0.8**, 10,000 walkers, 32 equilibration + 32 measurement blocks of length 0.2, eight forward cycles, and eight independent populations per disorder sample. At this time the largest descendant fraction was 0.1609, and the minimum Kish size 20.84 stayed above the floor 8. Overlap population-size dependence has not been directly compared; the earlier walker comparison tested energy.

The selected sample0 output is `production_observables.json`, with all eight populations listed in `production_input.json`: chi SG = 1.020229 ± 0.023033 SE, q EA squared = 0.009446566 ± 0.000213267 SE, q EA = 0.097193 (delta-method SE 0.001097). The susceptibility 95% interval is [0.965765, 1.074693]; the q EA interval is obtained by transforming the q2 interval. This one-sample output contains no measured disorder uncertainty. It is ready as the first sample at the tested setting; disorder production must retain the same outputs for each additional sample. No further simulation is currently running for this campaign.

## Required production outputs

`export_production.py` summarizes explicitly grouped independent population archives. For each disorder sample it preserves q, q2, q3 and q4, covariance of the population mean moments, the measurement-window energy, susceptibility, q_EA squared, the finite-size RMS q_EA, achieved times and completion diagnostics. At the ensemble level it takes equal disorder weights, computes q_EA from the mean q2, and reports population sampling uncertainty and disorder scatter separately. Sample-to-sample scatter contains projector noise; the additional disorder variance-component estimate subtracts the mean estimated population variance and clips negative results to zero. These are sampling errors; forward-time and finite-population biases require the convergence comparisons.

Create an input JSON with physical metadata and archive groups:

```json
{
  "n_sites": 108,
  "physical_settings": {"L": 6, "periodic": true, "sigma": 1, "gamma": 5, "h": 0},
  "samples": [
    {"sample_id": 0, "populations": ["runs/t2_p16.npz", "runs/t2_p17.npz"]}
  ]
}
```

Use all eight populations for the selected duration; the abbreviated example only illustrates the format. The input identifies the common physical problem and the disorder sample assigned to each population, because the raw `_project_once` fields do not contain coupling/sample metadata. Preserve campaign source hashes and physical metadata with any final production dataset.

```bash
python3 -B export_production.py production_input.json production_observables.json
```

Three analytic fixture tests in `test_export_production.py` passed, covering disorder averaging before the square root, distinct projector and disorder errors, single-sample missing disorder uncertainty, and rejection of duplicated archives. All 13 code cells in `notebook/gfmc_pqmc_imaginary_time.ipynb` executed successfully in a fresh kernel; the notebook was validated and saved with the results, formulas, exact harness and archive-based recomputation cells.
