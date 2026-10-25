"""One-sample L=6 trial of the forward-walked overlap.

The Friday figure is L=8. This run is the check that the projector, including
the 1/σ projection time, produces an overlap that falls as Γ/σ grows.

``--gate`` compares L=2 sample 0 with exact diagonalization.
``--run`` projects one (σ, Γ/σ) pair at L=6.
``--farm`` runs the nine trial pairs on a process pool.
``--table`` prints the pairs that have finished.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kagome_qea.exact import ground_state
from kagome_qea.pqmc import project
from kagome_qea.sample import make_sample

TRIAL_SIZE = 6
TRIAL_SAMPLE = 0
TRIAL_SIGMAS = (0.05, 1.0, 2.0)
TRIAL_RATIOS = (1.0, 2.0, 3.0)
PROJECTOR_SEED = 0
N_WALKERS = 5000
N_EQUILIBRATE = 40
N_MEASURE = 40
N_FORWARD_CYCLES = 24
TAU_BLOCK_AT_SIGMA_1 = 0.2
TAU_FORWARD_AT_SIGMA_1 = 4.0
GATE_SIGMAS = (1.0, 0.05)


def repo_root():
    return Path(__file__).resolve().parents[2]


def token(value):
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def output_path(sigma, gamma, size=TRIAL_SIZE):
    directory = (
        repo_root()
        / "data"
        / "kagome"
        / "gaussian_nn"
        / "pbc"
        / f"L{size}"
        / "pqmc"
        / "h0"
        / f"sigma{token(sigma)}"
        / f"gamma{token(gamma)}"
    )
    return directory / f"sample{TRIAL_SAMPLE}_seed{PROJECTOR_SEED}.npz"


def projector_times(sigma):
    """Block and forward times. Both scale as 1/σ, so the block count does not."""
    scale = 1.0 / float(sigma)
    return TAU_BLOCK_AT_SIGMA_1 * scale, TAU_FORWARD_AT_SIGMA_1 * scale


def project_pair(size, sigma, ratio):
    gamma = float(sigma) * float(ratio)
    tau_block, tau_forward = projector_times(sigma)
    sample = make_sample(size, True, TRIAL_SAMPLE)
    operator = sample.hamiltonian(0.0).with_classical_scale(sigma)
    started = time.perf_counter()
    result = project(
        operator,
        gamma,
        n_replicas=2,
        n_walkers=N_WALKERS,
        tau_block=tau_block,
        n_equilibrate=N_EQUILIBRATE,
        n_measure=N_MEASURE,
        tau_forward=tau_forward,
        n_forward_cycles=N_FORWARD_CYCLES,
        seed=PROJECTOR_SEED,
    )
    seconds = time.perf_counter() - started
    return sample, result, gamma, tau_block, tau_forward, seconds


def save_run(path, sample, result, sigma, gamma, tau_block, tau_forward, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        L=np.int64(sample.L),
        sample_id=np.int64(sample.sample_id),
        sigma=np.float64(sigma),
        gamma=np.float64(gamma),
        tau_block=np.float64(tau_block),
        tau_forward=np.float64(tau_forward),
        seconds=np.float64(seconds),
        seed_entropy=np.asarray(sample.seed_entropy, dtype=np.int64),
        J=np.asarray(sample.J, dtype=np.float64),
        q2=np.float64(result["q2"]),
        q2_error=np.float64(result["q2_error"]),
        q2_series=np.asarray(result["q2_series"], dtype=np.float64),
        q4=np.float64(result["q4"]),
        binder=np.float64(result["binder"]),
        energy=np.float64(result["energy"]),
        energy_error=np.float64(result["energy_error"]),
        energy_growth=np.float64(result["energy_growth"]),
        energy_growth_error=np.float64(result["energy_growth_error"]),
        min_unique_ancestors=np.int64(result["min_unique_ancestors"]),
    )


def run_one(sigma, ratio):
    path = output_path(sigma, sigma * ratio)
    if path.exists():
        print(f"skip sigma {sigma:g} ratio {ratio:g} {path}", flush=True)
        return str(path)
    sample, result, gamma, tau_block, tau_forward, seconds = project_pair(
        TRIAL_SIZE, sigma, ratio
    )
    save_run(path, sample, result, sigma, gamma, tau_block, tau_forward, seconds)
    print(
        f"L=6 sigma {sigma:g} ratio {ratio:g} gamma {gamma:.4f}  "
        f"q2 {result['q2']:.4f} ± {result['q2_error']:.4f}  "
        f"ancestors {result['min_unique_ancestors']}  "
        f"energy {result['energy']:.3f} growth {result['energy_growth']:.3f}  "
        f"{seconds:.0f}s",
        flush=True,
    )
    return str(path)


def gate():
    """Exact L=2 overlap at σ=1 and at the small-σ time scaling."""
    failed = False
    for sigma in GATE_SIGMAS:
        gamma = float(sigma)  # ratio 1
        sample, result, gamma, tau_block, tau_forward, seconds = project_pair(2, sigma, 1.0)
        exact = ground_state(sample.hamiltonian(0.0).with_classical_scale(sigma), gamma)
        difference = abs(result["q2"] - exact["q2"])
        limit = max(0.06, 3.0 * result["q2_error"])
        print(
            f"L=2 sigma {sigma:g} gamma {gamma:g}  {seconds:.1f}s  "
            f"tau_block {tau_block:g} tau_forward {tau_forward:g}\n"
            f"exact q2 {exact['q2']:.4f}  projector {result['q2']:.4f} ± {result['q2_error']:.4f}  "
            f"ancestors {result['min_unique_ancestors']}",
            flush=True,
        )
        if result["q2_error"] > 0.06 or difference > limit:
            print(f"FAIL difference {difference:.4f} limit {limit:.4f}", flush=True)
            failed = True
        else:
            print(f"pass difference {difference:.4f} limit {limit:.4f}", flush=True)
    if failed:
        raise SystemExit(1)


def farm(workers):
    jobs = [(sigma, ratio) for sigma in TRIAL_SIGMAS for ratio in TRIAL_RATIOS]
    workers = max(1, min(int(workers), len(jobs)))
    print(f"farm {len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, sigma, ratio) for sigma, ratio in jobs]
        for future in as_completed(futures):
            future.result()


def table():
    print(f"{'sigma':>6} {'ratio':>6} {'gamma':>8} {'q2':>8} {'err':>8} {'anc':>5} {'sec':>7}")
    missing = 0
    for sigma in TRIAL_SIGMAS:
        for ratio in TRIAL_RATIOS:
            path = output_path(sigma, sigma * ratio)
            if not path.exists():
                missing += 1
                print(f"{sigma:6g} {ratio:6g} {'missing':>8}")
                continue
            with np.load(path) as data:
                print(
                    f"{float(data['sigma']):6g} {ratio:6g} {float(data['gamma']):8.4f} "
                    f"{float(data['q2']):8.4f} {float(data['q2_error']):8.4f} "
                    f"{int(data['min_unique_ancestors']):5d} {float(data['seconds']):7.0f}"
                )
    if missing:
        raise SystemExit(f"{missing} trial files still missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--farm", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.gate:
        gate()
    elif args.run:
        if args.sigma is None or args.ratio is None:
            parser.error("--run needs --sigma and --ratio")
        run_one(args.sigma, args.ratio)
    elif args.farm:
        farm(args.workers)
    elif args.table:
        table()
    else:
        parser.error("choose --gate, --run, --farm, or --table")


if __name__ == "__main__":
    main()
