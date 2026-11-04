# L6 auxiliary-field VMC pilot artifacts

The current audited measurements and scientific limitations are in `RESULTS.md`
and `analysis_results.json`. The executed results notebook is
`../../../notebook/auxiliary_field_vmc_results.ipynb`.

- `candidate_bank.npz` preserves the classical candidates, selected references,
  graph and couplings; `candidate_bank.json` records their identities and hash.
- `training/` contains all seven branch histories, six saved model checkpoints
  per branch, and final resumable optimizer, sampler and RNG states.
- `frozen/` contains all 28 frozen diagnostics, with raw samples and energies
  in NPZ files and model/raw hashes in matching JSON records.
- `projection/` contains completed projection records, raw forward-walking
  measurements, reconstruction checks and progress records. The finite queue
  has fourteen runs; its current completion count is in `RESULTS.md`.
- `provenance/` preserves the initialization, immutable launched source,
  environment, launch records, logs, delivery manifest and local hash audit.
- `delivery_snapshot/20261104T030829Z/` preserves the older report files and
  provenance from the first complete VMC archive. It predates the current
  report and the restarted completion reporter.

The first VMC delivery archive and all 140 member hashes were verified before
copying. Its manifest is `provenance/delivery_vmc_complete_sha256.json`; the
mapping of copied, identical and separately preserved files is recorded in
`provenance/local_delivery_verification.json`. The archive includes complete
VMC data and only the projection measurements completed when it was created.
Later projection records retain their own raw-data and model hashes.

All scientific computation and notebook execution took place on AutoDL.
These artifacts cover one disorder and one training seed. Completion of the
finite queue does not establish ground-state or population/time convergence.
