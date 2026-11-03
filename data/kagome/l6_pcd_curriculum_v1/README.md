# L6 PCD/curriculum launch snapshot

`20261103T163115Z/` is a dated AutoDL snapshot, copied locally and verified
against its `sha256.json` inventory (21 files). Its validation record also
matches203 local source hashes. Scientific computation ran on AutoDL.

At 2026-11-03T16:31:15Z, PID396825 was running the first shared Γ=4 VMC
initialization, at update50/400. No new feedback stint or independent physical
measurement had completed. The executed notebook therefore reports pending
results. Its HTML is readable without a Python environment.

The existing L6 baseline/robust campaign had completed42/96 stints. The original
L6 pilot was measuring its first context's second repeat. L4, lower-field L6
anchors and higher-field L6 references were also alive. L8/L10 PIDs were absent;
their explicit stop records are included.

`validation/notebook_synthetic/` contains **synthetic display-test data only**;
it is not a physical result. Empty and populated notebook fixtures each executed
three code cells with zero errors. The numerical gate passed25 tests with zero
skips, failures or errors. The actual research notebook is under `results/`.

Remote live root: `/root/autodl-tmp/l6-pcd-curriculum-20261103-v1/`.
The worker updates `results/research_report.json` and executed notebook/HTML
after each disorder/seed dispatch group, with terminal status before the final
report. The local snapshot is static. Checkpoints, raw training/measurement
files, failures, ancestry alarms and source provenance remain on AutoDL.

At this checkpoint15.28GB remained free on the data disk. Existing L6 training
stint folders were about0.62MB each; the new sampler adds ladder artifacts.
Actual new-campaign growth and throughput can be assessed after completed
stints. An end-to-end completion estimate is not yet supported by measurements.

See `code/transformer_qmc/L6_PCD_CURRICULUM.md` for the full finite design,
literature basis, limitations and measurement protocol.
