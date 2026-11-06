# L6 D128 continuation

Remote campaign: `/root/autodl-tmp/triangle-L6-m6-c8-D128-dense-20261106-v1`.

The finite queue contains all 456 L6, m6, chi8 field/disorder cases. Each start
continues its own verified D64 MPS at D128, even if D64 was qualified. The
optimizer, physical model, disorder couplings, and numerical tolerances match
the D64 campaign. D64 files remain at their original remote paths.

The controller initially admits three representative cases to measure archive
growth. It admits the rest after those cases finish and the whole remaining
archive fits, including a 3 GiB reserve and a 25% projection margin. Numerical
failure of a qualification check does not prevent other cases from running.
Execution failures are recorded without automatic retries. There is no deadline.

Current local evidence is a snapshot, not a live mirror. Refresh from the
repository root with:

```sh
PYTHONPATH=code python3 -m transformer_qmc.fetch_l6_d128
```

This downloads completed-case JSON and NPZ evidence, verifies ancestry and
physical moments, then executes
`notebook/triangle_L6_m6_chi8_D128_convergence.ipynb`. Full gzip-compressed MPS
and optimizer checkpoints stay remote. Means require all eight disorders at a
field; both starts remain separate, with disorder-only standard errors.

Storage relocation is documented by `storage_relocation_inventory.json` and,
once complete, `storage_relocation_receipt.json` and
`storage_relocation_complete.json`. Older inactive project files are copied to
`data/kagome/autodl_relocated_for_D128_20261106/`; their notebooks go directly to
`notebook/autodl_relocated_for_D128_20261106/`. Every file is hash-verified locally
and again remotely before its remote copy is removed. Remote relocation markers
record the local destinations. Active files and the D64 campaign are excluded.
