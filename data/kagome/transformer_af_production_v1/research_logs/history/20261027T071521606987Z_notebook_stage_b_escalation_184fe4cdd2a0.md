# notebook_stage_b_escalation — research result log

Recorded UTC: 2026-10-27T07:15:21.606987+00:00

**System: L=4, N=48 spins, periodic boundaries in both directions.**

State: **waiting_for_base_pilot**. Completed results: **0/144**. Recorded failures: **0**.

Manifest signature: `0e910b5e5d6cfbe08460b7f5f45a37e0d34a8f928f8ae91958f730e37cccde7d`.

Process deadline UTC: 2026-10-27T17:46:47.218088+00:00.

All results are feasibility/development evidence. Ground-state convergence and phase-boundary inference remain unqualified. An ancestry alarm means any cycle has descendant ESS below 8 or largest-lineage fraction above 0.5. Every alarm is retained.

## Settings and completion

| Level | Gamma | h | Guide | W | Equilibration | Measurement | Forward | Cycles | Branch | Done / expected | Alarms | Median seconds | Minimum ESS |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_h005 | 0.5 | 0.05 | constant | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| baseline_h005 | 0.5 | 0.05 | network | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| baseline_h005 | 2.0 | 0.05 | constant | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| baseline_h005 | 2.0 | 0.05 | network | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 0.5 | 0.0 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 0.5 | 0.0 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 0.5 | 0.05 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 0.5 | 0.05 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 2.0 | 0.0 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 2.0 | 0.0 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 2.0 | 0.05 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 2.0 | 0.05 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 0.5 | 0.0 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 0.5 | 0.0 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 0.5 | 0.05 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 0.5 | 0.05 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 2.0 | 0.0 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 2.0 | 0.0 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 2.0 | 0.05 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 2.0 | 0.05 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 0.5 | 0.0 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 0.5 | 0.0 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 0.5 | 0.05 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 0.5 | 0.05 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 2.0 | 0.0 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 2.0 | 0.0 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 2.0 | 0.05 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 2.0 | 0.05 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 0.5 | 0.0 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 0.5 | 0.0 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 0.5 | 0.05 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 0.5 | 0.05 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 2.0 | 0.0 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 2.0 | 0.0 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 2.0 | 0.05 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population256 | 2.0 | 0.05 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |

## Individual population results

Values below are individual population estimates, without uncertainty bars. Forward cycles are correlated diagnostics, not independent repeats. Compare four independent repeats within each disorder before forming disorder averages. The escalation subset contains one disorder and cannot estimate disorder uncertainty.

| Task ID | Disorder | Repeat | Level | Gamma | h | Guide | E/N | q2 | chi_conn/N | Seconds | ESS minimum | Largest lineage | Alarm |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|

## Continuation and interpretation

The authorized next step is to finish the base pilot, then execute the fixed 144-run population/time escalation schedule. Its controller waits for successful base completion and acquires the shared supervisor lock. W256/W1024, separate equilibration4 and forward2 tests, and h=0.05 diagnose population loss, projection dependence and cost.

Ancestry recovery alone does not qualify physical observables. Stage C still requires measured feasible cost, eight development disorders, one-parameter convergence comparisons, alternative starts and combined checks. Any unresolved context remains unqualified. Final anchor scope and acceptance standards are identical for L=4,6,8,10.

At the recorded deadline workers checkpoint and pause; the instance stays on. Machine-readable source results and full cycle diagnostics are in `../notebook_stage_b_escalation/results/`. Snapshot provenance stores SHA-256 for every included result.
