"""Bounded energy-only stationarity study using the frozen flat PQMC solver."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE / "code"))
import kagome_qea.pqmc as pq
from kagome_qea.sample import make_sample

SETTINGS = dict(L=6, sample=0, pbc=True, sigma=1.0, h=0.0, gamma=0.05,
                n_walkers=5000, n_replicas=2, tau_block=0.2, n_blocks=2560,
                populations=8, seed=20261025, spawned_children=32,
                n_forward_cycles=0, window_blocks=320)

def write_json(path, obj):
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(target)

def save_trace(path, energy, times):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, energy=np.asarray(energy), time=np.asarray(times))
    temp.replace(path)

def run_population(child):
    start = time.monotonic()
    original = pq._branch_dt
    last_saved = 0
    def observed_interval(walkers, limit):
        nonlocal last_saved
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == "one_block" and frame.f_locals["covered"] == 0.0:
            energy = frame.f_locals["measure_energy"]
            times = frame.f_locals["measure_time"]
            count = len(energy)
            if count and count % SETTINGS["window_blocks"] == 0 and count != last_saved:
                save_trace(ROOT / "partial" / f"p{child}.npz", energy, times)
                write_json(ROOT / "partial" / f"p{child}.json",
                           dict(child=child, blocks=count, imaginary_time=float(times[-1]),
                                elapsed_seconds=time.monotonic()-start,
                                last_window_energy=float(np.mean(energy[-SETTINGS["window_blocks"]:]))))
                print(f"PROGRESS child={child} blocks={count} tau={times[-1]:.1f}", flush=True)
                last_saved = count
        del frame
        return original(walkers, limit)
    pq._branch_dt = observed_interval
    try:
        rng = np.random.default_rng(np.random.SeedSequence(SETTINGS["seed"]).spawn(
            SETTINGS["spawned_children"])[child])
        result = pq._project_once(make_sample(6, True, 0).hamiltonian(0.0), 0.05, rng,
            n_replicas=2, n_walkers=SETTINGS["n_walkers"], tau_block=0.2,
            n_equilibrate=0, n_measure=SETTINGS["n_blocks"], n_forward_cycles=0)
    finally:
        pq._branch_dt = original
    energy, times = result["energy_measure_trace"], result["energy_measure_time"]
    assert energy.shape == times.shape == (SETTINGS["n_blocks"],)
    assert np.all(np.isfinite(energy)) and np.allclose(times, np.arange(1, len(times)+1)*0.2)
    with (ROOT / "runs" / f"p{child}.npz").open("wb") as stream:
        np.savez_compressed(stream, **result)
    save_trace(ROOT / "partial" / f"p{child}.npz", energy, times)
    meta = dict(child=child, blocks=len(energy), imaginary_time=float(times[-1]),
                elapsed_seconds=time.monotonic()-start, finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / "runs" / f"p{child}.json", meta)
    return meta

def estimate(values):
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1)/np.sqrt(len(x)))
    half = float(student_t.ppf(0.975, len(x)-1))*se
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half], populations=len(x))

def difference(earlier, later, label):
    row = dict(label=label, **estimate(later-earlier))
    row["per_spin_mean"] = row["mean"]/108
    row["per_spin_ci95"] = [v/108 for v in row["ci95"]]
    row["resolved_drift"] = bool(row["ci95"][1] < 0 or row["ci95"][0] > 0)
    row["population_differences"] = (later-earlier).tolist()
    return row

def analyze():
    trajectories = []
    prefix_checks = []
    baseline = Path(json.loads((ROOT/"campaign.json").read_text())["baseline_remote"])
    for child in range(SETTINGS["populations"]):
        with np.load(ROOT/"runs"/f"p{child}.npz") as data:
            energy = data["energy_measure_trace"].copy()
            times = data["energy_measure_time"].copy()
        assert energy.shape == (SETTINGS["n_blocks"],) and np.all(np.isfinite(energy))
        trajectories.append(energy)
        for equil in (40,80,160):
            name = f"L6_eq{equil}_t8_p{child}.npz"
            with np.load(baseline/"runs"/name) as old:
                assert np.array_equal(energy[:equil],old["energy_equilibration_trace"]),name
                assert np.array_equal(energy[equil:equil+40],old["energy_measure_trace"]),name
            prefix_checks.append(dict(child=child, equilibration=equil, exact_match=True))
    matrix = np.stack(trajectories)
    windows = []
    means = []
    for start in range(0, SETTINGS["n_blocks"], SETTINGS["window_blocks"]):
        stop = start+SETTINGS["window_blocks"]
        population_means = matrix[:,start:stop].mean(axis=1)
        means.append(population_means)
        windows.append(dict(start_block=start, end_block=stop, start_time=start*0.2,
            end_time=stop*0.2, population_means=population_means.tolist(), **estimate(population_means)))
    shifts = [difference(means[i-1],means[i],f"window_{i}_to_{i+1}") for i in range(1,len(means))]
    first_late = matrix[:,1280:1920].mean(axis=1)
    last_late = matrix[:,1920:2560].mean(axis=1)
    primary = difference(first_late,last_late,"tau256_384_to_tau384_512")
    x = times[1280:]
    slopes = ((matrix[:,1280:]-matrix[:,1280:].mean(axis=1)[:,None])@(x-x.mean()))/np.sum((x-x.mean())**2)
    summary = dict(settings=SETTINGS,windows=windows,paired_window_shifts=shifts,
        primary_late_shift=primary, final_short_shift=shifts[-1],
        late_slope=estimate(slopes), population_late_slopes=slopes.tolist(),
        prefix_checks=prefix_checks,
        decision="resolved late-time energy drift" if primary["resolved_drift"] else
            "no primary late-time drift resolved at current precision; plateau not certified",
        ground_state_qualified=False, susceptibility_qualified=False,
        interpretation="Across-population intervals; energy stationarity is necessary evidence, not ground-state or susceptibility certification. No energy accuracy tolerance specified.",
        finished=datetime.now(timezone.utc).isoformat(), failures=[])
    write_json(ROOT/"final.json",summary)
    write_report(summary)
    plot(matrix,times)
    return summary

def write_report(s):
    primary=s["primary_late_shift"]
    out=["# Extended L6 energy-stability check","",f"**Decision: {s['decision']}. Ground-state qualification remains false.**","",
        "Flat sampler; L6/sample0 (108 spins), Gamma=0.05, 5,000 walkers, two replicas, eight independent populations. Each trajectory covers 2,560 blocks, total imaginary time 512. All early blocks are diagnostics, not equilibrated measurements.","",
        "## Energy versus imaginary time","",
        "Intervals below are 95% Student-t intervals across eight population means; energies are total single-replica energies.","",
        "| Time interval | Energy mean | SE | 95% interval |","| --- | ---: | ---: | --- |"]
    for w in s["windows"]:
        out.append(f"| {w['start_time']:.0f}–{w['end_time']:.0f} | {w['mean']:.6f} | {w['se']:.6f} | [{w['ci95'][0]:.6f}, {w['ci95'][1]:.6f}] |")
    out += ["","## Primary late-window comparison","",
        f"Time 256–384 versus 384–512: paired energy change **{primary['mean']:.6f} +/- {primary['se']:.6f} SE**, 95% interval [{primary['ci95'][0]:.6f}, {primary['ci95'][1]:.6f}].",
        f"Per spin: change {primary['per_spin_mean']:.8f}, 95% interval [{primary['per_spin_ci95'][0]:.8f}, {primary['per_spin_ci95'][1]:.8f}].","",
        "| Population | Energy, time 256–384 | Energy, time 384–512 | Paired change |","| --- | ---: | ---: | ---: |"]
    for child in range(8):
        early=np.mean(s["windows"][4]["population_means"][child:child+1]+s["windows"][5]["population_means"][child:child+1])
        late=np.mean(s["windows"][6]["population_means"][child:child+1]+s["windows"][7]["population_means"][child:child+1])
        out.append(f"| {child} | {early:.6f} | {late:.6f} | {late-early:.6f} |")
    last=s["final_short_shift"]; slope=s["late_slope"]
    out += ["",f"Final short-window change (time 384–448 to 448–512): {last['mean']:.6f} +/- {last['se']:.6f} SE, 95% interval [{last['ci95'][0]:.6f}, {last['ci95'][1]:.6f}].",
        f"Slope over time 256–512: {slope['mean']:.8f} total-energy units per imaginary-time unit; 95% interval [{slope['ci95'][0]:.8f}, {slope['ci95'][1]:.8f}].","",
        "## Interpretation and provenance","",
        "An interval containing zero means no drift was resolved at this precision; it is not an equivalence test. No absolute/relative energy tolerance has been specified. Population disagreement, finite-walker bias and possible trapping remain relevant even if the energy looks stationary. These runs do not extend the pure-overlap forward-walking horizon and do not qualify susceptibility.","",
        f"All {len(s['prefix_checks'])} matched eq40/80/160 trajectory-prefix checks reproduce the previous pilot exactly. Frozen source identity and settings are in `campaign.json`; partial and complete traces are retained. No solver source change was required for this study. Production remains on hold.","",
        "See `energy_trace.svg`, `final.json`, `PROTOCOL.md`, and `runs/`.",""]
    (ROOT/"REPORT.md").write_text("\n".join(out))

def plot(matrix,times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    width=40
    x=times.reshape(-1,width).mean(axis=1)
    y=matrix.reshape(8,-1,width).mean(axis=2)
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for row in y:
        axes[0].plot(x,row,alpha=0.5,linewidth=0.8)
        axes[1].plot(x,row/108,alpha=0.5,linewidth=0.8)
    axes[0].plot(x,y.mean(axis=0),color="black",linewidth=2,label="8-population mean")
    axes[1].plot(x,y.mean(axis=0)/108,color="black",linewidth=2)
    for ax in axes:
        ax.axvline(256,color="gray",linestyle="--")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Total energy (one replica)")
    axes[1].set_ylabel("Energy per spin")
    axes[1].set_xlabel("Cumulative imaginary time")
    axes[0].legend()
    fig.suptitle("L6/sample0, flat PQMC, Gamma=0.05, 5000 walkers\n40-block smoothing; full unsmoothed traces retained")
    fig.tight_layout()
    fig.savefig(ROOT/"energy_trace.svg")
    fig.savefig(ROOT/"energy_trace.png",dpi=150)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    start=time.monotonic()
    campaign=json.loads((ROOT/"campaign.json").read_text())
    for name,digest in campaign["source_sha256"].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest,name
    assert campaign["settings"]==SETTINGS
    for name in ("runs","partial"):(ROOT/name).mkdir(exist_ok=True)
    if not args.analyze_only:
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(run_population,child) for child in range(8)]
            for future in as_completed(futures):
                row=future.result()
                print("DONE",json.dumps(row),flush=True)
    final=analyze()
    final["analysis_elapsed_seconds" if args.analyze_only else "campaign_elapsed_seconds"]=time.monotonic()-start
    write_json(ROOT/"final.json",final)
    print("FINAL",json.dumps(dict(decision=final["decision"],primary_late_shift=final["primary_late_shift"],elapsed_seconds=time.monotonic()-start)),flush=True)

if __name__=="__main__":
    try:
        main()
    except BaseException:
        (ROOT/"failure.txt").write_text(traceback.format_exc())
        raise
