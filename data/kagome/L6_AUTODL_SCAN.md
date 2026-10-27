# L=6 AutoDL scan

Started 2026-10-25. Parent PID 736915 on `autodl` (`connect.westc.seetacloud.com`, port 23649, user root). The process is detached with `setsid`. Closing the laptop does not stop it. Do not shut the instance down.

## What is running

`code/kagome_qea/trial_overlap.py --queue --workers 16`

Python: `/root/miniconda3/envs/yjye_project_env/bin/python -u`

One queue holds every run. A file already on disk is skipped, so starting the same command again continues the queue. There is no second handoff. Periodic Kagome, \(L = 6\), \(N = 108\), \(h = 0\). Four draws, `sample_id = 0, 1, 2, 3`. Six standard deviations, \(\sigma = 0.05, 0.1, 0.2, 0.5, 1, 2\), on the same stored \(z_b\) via `with_classical_scale`. Dense ratios

```
0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0
```

360 projector runs. Finished files are skipped. Projector: 5000 walkers, 2 replicas, 40 discard, 40 measure, 24 forward cycles, \(\tau_{\mathrm{block}} = 0.2/\sigma\), \(\tau_{\mathrm{forward}} = 4/\sigma\), seed 0.

About three hours on 16 cores from an empty disk, less the files already saved. Each finished run is written to its `.npz` and then copied to `saved/` before the worker is free for the next run. A half-written file is not kept and is not skipped. The process draws the figure itself when the queue is empty. Two cores stay for the daily updater. The manifest is `l6_queue.txt`. The one-line count is `l6_queue_status.txt`.

Live files:

```
/root/autodl-tmp/kagome-qea/data/kagome/gaussian_nn/pbc/L6/pqmc/
/root/autodl-tmp/kagome-qea/saved/data/kagome/gaussian_nn/pbc/L6/pqmc/
```

The second tree is a copy made at the moment each file is closed. A local copy of whatever has already finished is under `data/kagome/gaussian_nn/pbc/L6/pqmc/` in this repo. Copy again with the rsync in the section below.

## Where to look

Remote log:

```
/root/autodl-tmp/kagome-qea/l6_scan.log
```

Each finished run prints one line: `L=6 sigma ... ratio ... sample ... q2 ... ancestors ...`. A skip prints `skip sigma ...`.

Results, remote and local after copy-back:

```
data/kagome/gaussian_nn/pbc/L6/pqmc/h0/sigma{sigma}/gamma{gamma}/sample{id}_seed0.npz
```

Remote root is `/root/autodl-tmp/kagome-qea/`. Keys in each file: `q2`, `q2_error`, `q2_series`, `q4`, `binder`, `energy`, `energy_growth`, `min_unique_ancestors`, `sigma`, `gamma`, `J`, `seconds`.

## Check, copy back, resume

```bash
export SSHPASS="$(security find-generic-password -s autodl-password -w)"
sshpass -e ssh autodl "pgrep -af 'trial_overlap.py --queue'; echo '---'; cat /root/autodl-tmp/kagome-qea/l6_queue_status.txt; echo '---'; tail -n 15 /root/autodl-tmp/kagome-qea/l6_scan.log"
```

Done when the parent PID is gone and the log has no traceback. Copy back:

```bash
rsync -az -e "sshpass -e ssh" \
  autodl:/root/autodl-tmp/kagome-qea/data/kagome/gaussian_nn/pbc/L6/pqmc/ \
  data/kagome/gaussian_nn/pbc/L6/pqmc/
rsync -az -e "sshpass -e ssh" \
  autodl:/root/autodl-tmp/kagome-qea/l6_scan.log \
  data/kagome/l6_scan.log
```

Resume is the same `--queue` command. It skips any `sample{id}_seed0.npz` already on disk.

## Figure

The queue process writes the figure itself when every file is present. The figure and the table are:

```
/root/autodl-tmp/kagome-qea/l6_scan.png
/root/autodl-tmp/kagome-qea/l6_scan_table.txt
/root/autodl-tmp/kagome-qea/l6_plot.log
```

`PLOT_FINISHED` in `l6_plot.log` means the PNG exists. Copy them back with the results:

```bash
rsync -az -e "sshpass -e ssh" \
  autodl:/root/autodl-tmp/kagome-qea/l6_scan.png \
  autodl:/root/autodl-tmp/kagome-qea/l6_scan_table.txt \
  autodl:/root/autodl-tmp/kagome-qea/l6_plot.log \
  data/kagome/
```

The PNG has four panels: $[\langle q^2\rangle]$ and $[B]$ against $\Gamma$, and the same two against $\Gamma/\sigma$. The dashed line is $1/N$.

## Known before this launch

Sample 0 at \(\sigma = 1\) with 5000 walkers gave \(q_2 \approx 0.004\) at \(\Gamma = 0.05\), \(0.037\) at \(0.2\), and \(0.023\) at \(0.4\), with one or two ancestral lines. 20000 walkers at \(\Gamma = 0.2\) stayed at \(q_2 = 0.022\) and one line. \(1/N = 0.009\). The scan was started anyway, at 5000 walkers, so the dense curve exists. Read `min_unique_ancestors` before treating \(q_2\) as a transition.
