# Matched L6 chi8 fallback scan

Authorized scope: L=6, m=6, chi=8, Gamma=0.100:0.025:1.500,
eight matched disorders 60000–60007, 456 cases. The user authorized chi8
if the same chi16 scan was not feasible on the current allocation.

At 2026-11-05 04:31:58 UTC, chi8 was selected from measured storage.
The chi16 model plus its first D32 stage alone occupied 12,023,360 bytes.
Repeating that partial footprint across 456 cases is 5.106 GiB, above the
4.433 GiB available after preserving 3 GiB free. This is an incomplete
archive projection, not a completed-case cost estimate. The chi16 pilot was
stopped after exact argv verification and all saved artifacts were verified
unchanged. Its L4 exact validation had passed.

The chi8 L4 exact validation also passed. Four L6 feasibility cases cover
(Gamma,disorder)=(0.1,60000),(1.0,60003),(1.5,60000),(1.0,60000).
The controller adopts existing workers, requires all four pilots to qualify,
and checks measured archive growth before expanding to the full grid.
The prospective disk projection uses 1.25 times the largest completed
archive, plus active checkpoint reservations and 3 GiB free.

The existing two-start D32/64/128/256/512 solver is unchanged. Original
energy, spectrum, canonical, truncation and R gates remain active, together
with qEA absolute and connected susceptibility/raw q2 relative stability
checks. Unresolved cases remain unresolved. Whole-disorder uncertainty
requires eight qualified disorders; individual curves have no error bars.

Remote campaign:
`/root/autodl-tmp/triangle-L6-m6-c8-dense-20261105-v1/`

Remote chi16 feasibility evidence:
`/root/autodl-tmp/triangle-L6-m6-c16-dense-20261105-v1/`

Keep every raw result, model, checkpoint and failure remote. Fetch only small
evidence here. Any notebook is delivered directly to
`notebook/triangle_L6_m6_chi8_dense.ipynb`, never staged in `data/`.

The earlier three-size/high-retention campaigns remain stopped. The unrelated
overlap campaign remains running. There is no local persistent fetcher.

## Superseded by user-authorized D64 restart

On 2026-11-05 at 05:13:21 UTC, the old controller and three D256 workers were
stopped for the user's requested D64 cap. All case files remain remote and
were hash-verified unchanged. The active replacement is
`/root/autodl-tmp/triangle-L6-m6-c8-D64-dense-20261105-v1`, with local evidence
in `../triangle_L6_m6_chi8_D64_dense_v1/`. It reuses saved D32/D64 stages and
executes the same 456 matched cases, retaining strict numerical flags without
escalating above D64. Earlier status snapshots in this directory are historical.

## User-authorized storage cleanup, 2026-11-05

Verified at 05:01:29 UTC: data-disk free space rose from 7.630 to 9.350 GiB
(net +1.720 GiB during live computation). System-disk free space rose from
5.886 to 6.259 GiB (+0.373 GiB). Evidence: `cleanup/summary.json`.

- Removed disposable pip download/wheel/self-check cache, not installed packages.
- Losslessly compressed the full historical export archive from
  `/root/autodl-tmp/alphaqubit-aq2/export_onnx/models.tar` to `models.tar.gz`.
  All 90 archived versions are retained, including 30 differing from unpacked
  models. Decompression reproduced the complete original SHA256.
- Losslessly compressed 91 closed historical training logs to adjacent `.gz`
  paths, retaining their content and timestamps. Every roundtrip was verified.
- Verified and shared duplicate blocks for 685 inactive file destinations.
  This produced no measured capacity gain, so no reclaimed space is credited.

All 456 chi3 baseline case metadata/raw hashes and completed chi8 case hashes
were rechecked after cleanup. The three remaining chi8 pilot workers and
their controller remain active. No scientific result or checkpoint was deleted.
Detailed per-file operations stay remote under the campaign's `cleanup/`.
