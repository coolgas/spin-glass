"""Generator-matched, equal-quota geometry QFI using batched exact dynamics.

No inference model is used. All positions are in micrometers, hbar=1.
Outputs per-realization QFI/leakage on a common time grid and bin summaries.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.special import ndtr, ndtri


def bins(count):
    edges = ndtri(ndtr(0.5) + np.arange(count + 1) / count * (1 - ndtr(0.5)))
    edges[0], edges[-1] = 0.5, np.inf
    return edges


def sample_site(length, site, edges, quota, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(len(edges) - 1)]
    counts = np.zeros(len(buckets), dtype=int)
    draws = rejected = overflow = 0
    while np.any(counts < quota):
        if draws >= 10_000_000:
            raise RuntimeError(f'Quota incomplete: L={length}, site={site}, counts={counts}')
        noise = rng.normal(0, 0.02, (4096, length))
        a = rng.normal(0, 1, 4096)
        noise[:, site] = a
        for row, displacement in zip(noise, a):
            draws += 1
            if abs(displacement) <= 0.5 or np.any(np.diff(np.arange(length) * 8 + row) <= 0):
                rejected += 1
                continue
            m = np.searchsorted(edges, abs(displacement), side='right') - 1
            if counts[m] == quota:
                overflow += 1
                continue
            buckets[m].append(row.copy())
            counts[m] += 1
            if np.all(counts == quota):
                break
    return np.asarray(buckets), dict(draws=draws, rejected=rejected, overflow=overflow)


class Solver:
    def __init__(self, length, device):
        self.length, self.device = length, device
        self.basis_np = ((np.arange(2**length)[:, None] >> np.arange(length - 1, -1, -1)) & 1)
        basis = torch.tensor(self.basis_np, dtype=torch.float64, device=device)
        self.pairs = torch.triu_indices(length, length, offset=1, device=device)
        self.occupied_pairs = basis[:, self.pairs[0]] * basis[:, self.pairs[1]]
        dim = len(basis)
        self.drive = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        ids = torch.arange(dim, device=device)
        for j in range(length):
            self.drive[ids, ids ^ (1 << j)] = 2.5 / 2
        self.nominal = torch.arange(length, dtype=torch.float64, device=device) * 8
        self.outside = torch.tensor(np.any(self.basis_np[:, :-1] * self.basis_np[:, 1:], axis=1), device=device)

    def hamiltonian(self, noise, c6):
        x = self.nominal + noise
        distance = x[:, self.pairs[1]] - x[:, self.pairs[0]]
        if torch.any(distance <= 0):
            raise ValueError('Atom order lost')
        diagonal = (c6 / distance**6) @ self.occupied_pairs.T
        return self.drive[None] + torch.diag_embed(diagonal)

    def spectrum(self, noise, c6):
        w, v = torch.linalg.eigh(self.hamiltonian(noise, c6))
        return w, v.to(torch.complex128), v[:, 0, :].to(torch.complex128)

    @staticmethod
    def states(spec, ts):
        w, v, coeff = spec
        phase = torch.exp(-1j * w[:, :, None] * ts[None, None, :])
        return v @ (phase * coeff[:, :, None])

    def evaluate(self, noise, site, c6, times, step=5e-5, chunk=200):
        noise = torch.as_tensor(noise, dtype=torch.float64, device=self.device)
        plus, minus = noise.clone(), noise.clone()
        plus[:, site] += step
        minus[:, site] -= step
        base = self.spectrum(noise, c6)
        positive = self.spectrum(plus, c6)
        negative = self.spectrum(minus, c6)
        qs, ls = [], []
        max_norm_error = 0.0
        for offset in range(0, len(times), chunk):
            ts = torch.tensor(times[offset:offset + chunk], device=self.device, dtype=torch.float64)
            psi = self.states(base, ts)
            derivative = (self.states(positive, ts) - self.states(negative, ts)) / (2 * step)
            probabilities = psi.abs().square()
            qfi = 4 * (derivative.abs().square().sum(1) - (psi.conj() * derivative).sum(1).abs().square())
            leakage = probabilities[:, self.outside].sum(1)
            max_norm_error = max(max_norm_error, float((probabilities.sum(1) - 1).abs().max()))
            qs.append(qfi.cpu().numpy())
            ls.append(leakage.cpu().numpy())
        q, leak = np.concatenate(qs, 1), np.concatenate(ls, 1)
        if not np.isfinite(q).all() or not np.isfinite(leak).all():
            raise RuntimeError('Nonfinite observable')
        if q.min() < -1e-7 or leak.min() < -1e-10 or leak.max() > 1 + 1e-10 or max_norm_error > 1e-9:
            raise RuntimeError('Physical consistency failure')
        if times[0] == 0 and (np.max(np.abs(q[:, 0])) > 1e-10 or np.max(np.abs(leak[:, 0])) > 1e-10):
            raise RuntimeError('Nonzero initial observables')
        return np.maximum(q, 0), leak, max_norm_error


def validate(device):
    from scipy.linalg import expm, expm_frechet
    from src.model import RydSimFull1D
    records = []
    for length in [6, 7]:
        solver = Solver(length, device)
        for site, displacement in [(0, -2.8), (length // 2, 1.0), (length - 1, 2.8)]:
            noise = np.random.default_rng(89 + site).normal(0, .02, length)
            noise[site] = displacement
            c6 = 2_000_000
            reference, _ = RydSimFull1D(length, 2.5, 0, c6, 8).build_Ham(noise)
            actual = solver.hamiltonian(torch.tensor(noise[None], device=device), c6)[0].cpu().numpy()
            np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-13)
            positions = np.arange(length) * 8 + noise
            d_diag = np.zeros(2**length)
            for i in range(length):
                for j in range(i + 1, length):
                    d_diag += (-6 * c6 / (positions[j] - positions[i])**7 * ((j == site) - (i == site))) * solver.basis_np[:, i] * solver.basis_np[:, j]
            t = 19.99
            unitary, tangent = expm_frechet(-1j * reference * t, -1j * np.diag(d_diag) * t)
            psi, derivative = unitary[:, 0], tangent[:, 0]
            expected = 4 * (np.vdot(derivative, derivative).real - abs(np.vdot(psi, derivative))**2)
            ts = np.array([0., 1., t])
            q, leak, norm = solver.evaluate(noise[None], site, c6, ts)
            q_half, _, _ = solver.evaluate(noise[None], site, c6, ts, step=2.5e-5)
            np.testing.assert_allclose(q[0, -1], expected, rtol=2e-4, atol=1e-7)
            np.testing.assert_allclose(q, q_half, rtol=2e-4, atol=1e-7)
            expected_leak = np.abs(expm(-1j * reference * t)[:, 0][solver.outside.cpu().numpy()])**2
            np.testing.assert_allclose(leak[0, -1], expected_leak.sum(), atol=1e-10)
            records.append(dict(length=length, site=site, qfi=float(expected), relative_error=float(abs(q[0, -1] - expected) / max(expected, 1e-12)), norm_error=norm))
    return records


def summarize(out):
    """Small portable summaries and explicitly labelled common-time previews."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    summary = {}
    rows = []
    coefficients = set()
    for path in sorted(out.glob('L*_site*.h5')):
        with h5py.File(path, 'r') as f:
            if not f.attrs.get('complete', False):
                continue
            key = path.stem
            coefficients.add(float(f.attrs['c6']))
            summary[key + '_spread'] = f['signed_std'][:]
            summary[key + '_mean_qfi'] = f['mean_qfi'][:]
            summary[key + '_mean_leakage'] = f['mean_leakage'][:]
            summary[key + '_positive_count'] = f['positive_count'][:]
            summary['times'] = f['times'][:]
            summary['bin_edges'] = f['bin_edges'][:]
            for index in [100, 500, 1000, 1999]:
                for m in range(len(f['signed_std'])):
                    rows.append([int(f.attrs['length']), int(f.attrs['site']), m, float(f['times'][index]), int(f['count'][m]), float(f['signed_std'][m]), float(f['mean_qfi'][m,index]), float(f['mean_leakage'][m,index])])
    np.savez_compressed(out/'summary.npz', **summary)
    with (out/'common_time_previews.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['L','site','bin','time','count','signed_std_um','mean_qfi_per_um2','mean_leakage'])
        writer.writerows(rows)
    for index in [100, 500, 1000, 1999]:
        fig, axes = plt.subplots(2,2,figsize=(12,8), constrained_layout=True)
        for col, length in enumerate([6,7]):
            for site in range(length):
                key = f'L{length}_site{site}'
                if key+'_spread' not in summary:
                    continue
                x = summary[key+'_spread']
                order = np.argsort(x)
                for row, observable in enumerate(['qfi','leakage']):
                    axes[row,col].plot(x[order], summary[key+'_mean_'+observable][order,index], '.-', label=f'site {site}')
                    axes[row,col].set_xlabel('Within-bin signed displacement std (µm)')
                    axes[row,col].set_ylabel('Mean displacement QFI (µm⁻²)' if row==0 else 'Mean global blockade leakage')
                    axes[row,col].set_title(f'L={length}')
                    axes[row,col].grid(alpha=.2)
            axes[0,col].legend(fontsize=8)
        fig.suptitle(f'Common-time preview t={summary["times"][index]:g}, C6={sorted(coefficients)}; generator-matched bins')
        fig.savefig(out/f'preview_t{summary["times"][index]:g}.png',dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['benchmark', 'run'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--quota', type=int, default=100)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--c6', type=float, default=2_000_000)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    times = np.arange(2000, dtype=float) * .01
    edges = bins(args.bins)
    if args.mode == 'benchmark':
        report = dict(validation=validate(args.device), device=args.device, batches=[])
        for length in [6, 7]:
            solver = Solver(length, args.device)
            geometries, _ = sample_site(length, 0, edges, 2, args.seed)
            for batch in [8, 32, 64]:
                noise = np.resize(geometries.reshape(-1, length), (batch, length))
                start = time.perf_counter()
                solver.evaluate(noise, 0, args.c6, times)
                elapsed = time.perf_counter() - start
                row = dict(length=length, batch=batch, seconds=elapsed, per_geometry=elapsed/batch)
                report['batches'].append(row)
                print(json.dumps(row), flush=True)
        (args.out / 'benchmark.json').write_text(json.dumps(report, indent=2))
        print('BENCHMARK_COMPLETE', flush=True)
        return
    config = vars(args).copy()
    config['out'] = str(args.out)
    config.update(provenance='fresh generator-matched data; original Ada training coverage unverified', time_grid='0:0.01:19.99', derivative_step=5e-5, blockade='no adjacent excitations', initial_state='all ground', background_std=.02, defect_std=1., cutoff=.5, lengths=[6,7])
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    start_run = time.perf_counter()
    for length in [6,7]:
        solver = Solver(length, args.device)
        for site in range(length):
            path = args.out / f'L{length}_site{site}.h5'
            if path.exists():
                with h5py.File(path, 'r') as f:
                    if f.attrs.get('complete', False):
                        print(f'SKIP complete {path.name}', flush=True)
                        continue
                raise RuntimeError(f'Incomplete output exists: {path}; use a new output directory')
            geometries, stats = sample_site(length, site, edges, args.quota, args.seed + length * 100 + site)
            assert geometries.shape == (args.bins, args.quota, length)
            # Check the largest displacement of each sign in every bin at the
            # latest time before selecting one step for the entire site's run.
            probes = np.concatenate([geometries[m, [np.argmin(geometries[m,:,site]), np.argmax(geometries[m,:,site])]] for m in range(args.bins)])
            step = 5e-5
            for attempt in range(8):
                q1, _, _ = solver.evaluate(probes, site, args.c6, np.array([1., 10., 19.99]), step=step)
                q2, _, _ = solver.evaluate(probes, site, args.c6, np.array([1., 10., 19.99]), step=step/2)
                if np.all(np.abs(q1-q2) <= 1e-7 + 2e-4*np.abs(q2)):
                    break
                step /= 2
            else:
                raise RuntimeError('Finite-difference convergence failed for sampled tail geometries')
            print(f'CHECK L={length} site={site} step={step} counts={args.quota} per bin', flush=True)
            with h5py.File(path, 'w') as f:
                f.attrs.update(length=length, site=site, c6=args.c6, complete=False, sampling=json.dumps(stats), derivative_step=step)
                f.create_dataset('positions_offset', data=geometries)
                f.create_dataset('bin_edges', data=edges)
                f.create_dataset('times', data=times)
                f.create_dataset('count', data=np.full(args.bins, args.quota))
                a = geometries[:,:,site]
                f.create_dataset('signed_mean', data=a.mean(1))
                f.create_dataset('signed_std', data=a.std(1, ddof=1))
                f.create_dataset('positive_count', data=(a > 0).sum(1))
                q_out = f.create_dataset('qfi', (args.bins,args.quota,len(times)), dtype='f8', chunks=(1,min(10,args.quota),len(times)), compression='lzf')
                l_out = f.create_dataset('leakage', q_out.shape, dtype='f8', chunks=q_out.chunks, compression='lzf')
                for m in range(args.bins):
                    for offset in range(0, args.quota, args.batch):
                        q, leak, norm = solver.evaluate(geometries[m,offset:offset+args.batch], site, args.c6, times, step=step)
                        q_out[m,offset:offset+len(q)] = q
                        l_out[m,offset:offset+len(q)] = leak
                    f.flush()
                    print(json.dumps(dict(length=length,site=site,bin=m+1,bins=args.bins,elapsed_seconds=time.perf_counter()-start_run)), flush=True)
                f.create_dataset('mean_qfi', data=np.stack([q_out[m].mean(0) for m in range(args.bins)]))
                f.create_dataset('mean_leakage', data=np.stack([l_out[m].mean(0) for m in range(args.bins)]))
                f.attrs['complete'] = True
            print(f'COMPLETE {path.name}', flush=True)
    summarize(args.out)
    print(f'RUN_COMPLETE seconds={time.perf_counter()-start_run:.3f}', flush=True)


if __name__ == '__main__':
    main()
