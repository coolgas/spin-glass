# OBC shared-guide training campaign

State: **main_running**

Separate OBC-v1 architecture and checkpoint namespace; all runs use open kagome patches.
Recipe: fresh 600-step batch-128 pilot, then 6000-step batch-256 main initialized from pilot.
Sizes 2,4,6,8 contribute equally to every VMC step; fields (0.5,0), (0.5,0.3), (2,0.3).
Exact references are evaluation-only. Training completion does not qualify GFMC convergence.

## smoke_full

```json
{
  "state": "completed",
  "phase": "smoke",
  "boundary": "obc",
  "updated_unix": 1793176631.3810658,
  "total": 3,
  "step": 3,
  "checkpoint": "obc_v1_shared_alibi__smoke__L02-04-06-08__seed501__final.pt",
  "sha256": "b65d5f5247c16da71b39bbdf40625f620ead95e7144eca8ce5f727180f88bd90",
  "elapsed_seconds": 2.0483135655522346,
  "peak_memory_mib": 337.88720703125
}
```

## smoke_resume

```json
{
  "state": "completed",
  "phase": "smoke",
  "boundary": "obc",
  "updated_unix": 1793176642.861572,
  "total": 3,
  "step": 3,
  "checkpoint": "obc_v1_shared_alibi__smoke__L02-04-06-08__seed501__final.pt",
  "sha256": "b65d5f5247c16da71b39bbdf40625f620ead95e7144eca8ce5f727180f88bd90",
  "elapsed_seconds": 2.5710082314908504,
  "peak_memory_mib": 337.6494140625
}
```

## capacity_128

```json
{
  "state": "completed",
  "phase": "smoke",
  "boundary": "obc",
  "updated_unix": 1793176650.4033952,
  "total": 3,
  "step": 3,
  "checkpoint": "obc_v1_shared_alibi__smoke__L02-04-06-08__seed501__final.pt",
  "sha256": "0035485f752a6cd9d2f7c3d358c9b0f948af856bb41d2579def944816ca35d14",
  "elapsed_seconds": 3.2207837495952845,
  "peak_memory_mib": 461.90234375
}
```

## capacity_256

```json
{
  "state": "completed",
  "phase": "smoke",
  "boundary": "obc",
  "updated_unix": 1793176659.261509,
  "total": 3,
  "step": 3,
  "checkpoint": "obc_v1_shared_alibi__smoke__L02-04-06-08__seed501__final.pt",
  "sha256": "d7d66f9a91569e27d6ba5f1559244257edafb775e2d14f45c0759db9bc02e7b8",
  "elapsed_seconds": 4.604648392647505,
  "peak_memory_mib": 498.1865234375
}
```

## pilot

```json
{
  "state": "completed",
  "phase": "pilot",
  "boundary": "obc",
  "updated_unix": 1793177338.5252466,
  "total": 600,
  "step": 600,
  "checkpoint": "obc_v1_shared_alibi__pilot__L02-04-06-08__seed501__final.pt",
  "sha256": "6bade0261b6e17731ed0169f02b59592ff02532e2b1ca9d068cc4062d9a54ac3",
  "elapsed_seconds": 675.126152202487,
  "peak_memory_mib": 463.78369140625
}
```

## main

```json
{
  "state": "training",
  "phase": "main",
  "boundary": "obc",
  "updated_unix": 1793179665.9223065,
  "total": 6000,
  "step": 1770,
  "elapsed_seconds": 2323.054119599983,
  "peak_memory_mib": 500.1650390625
}
```

## Supervisor details

```json
{
  "child_pid": 893450
}
```
