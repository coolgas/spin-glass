# AutoDL disk cleanup, 2026-11-03

Completed at 23:24:50 UTC. Free space increased from 14,893,842,432 to
17,756,344,320 bytes: a net gain of 2,862,501,888 bytes (2.86 GB).

- Removed 3,053 redundant files from the preliminary notebook bundle only after
  matching each file's size and SHA-256 against a retained local copy.
- Archived four inactive historical raw-result directories using lossless gzip
  tar archives. Verified every archive member against the original SHA-256 and
  checked that original files had not changed before removing loose copies.
- Retained active sources, checkpoints, raw measurements, failures and ancestry
  diagnostics. Unique HDF5 data and unrelated projects were retained.
- The historical hardlink candidates in the plan were not applied.

The report records all removed duplicate paths and retained local copies. Each
archive manifest records its remote path, archive hash, and every member hash.
The archives remain on AutoDL beside their original directories. Each original
directory contains `ARCHIVED_RESULTS.md` with its restoration command. Historical
analysis that expects loose JSON files requires extraction first.

All five checked campaign workers were alive after cleanup. The stopped L8 and
L10 workers remained absent. The live PCD campaign reported no execution failures.

The first multi-source SCP download returned exit code 5 after transferring the
report. A subsequent SSH tar transfer successfully retrieved the full report,
plan and all four manifests into this directory.
