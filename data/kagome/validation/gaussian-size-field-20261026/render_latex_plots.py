"""Render publication-style scan figures with mathtext on every axis label."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    result = json.loads((ROOT/'final.json').read_text())
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 13,
        'legend.fontsize': 9,
        'mathtext.fontset': 'dejavusans',
    })
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    series = [
        ('chi_sg', r'$\chi_{\mathrm{SG}}$', True),
        ('q_ea', r'$q_{\mathrm{EA}}$', True),
        ('binder_ratio_of_disorder_moments', r'$U_4$', False),
    ]
    for L in (4, 6, 8):
        rows = [row for row in result['curves'] if row['L'] == L]
        x = np.array([row['gamma_over_sigma'] for row in rows])
        for ax, (name, ylabel, with_error) in zip(axes, series):
            y = np.array([row[name] for row in rows])
            if with_error and name == 'chi_sg':
                error = np.array([3*L*L*row['q2_sample_scatter_se'] for row in rows])
                ax.errorbar(x, y, yerr=error, marker='o', capsize=3, lw=1.7,
                            label=fr'$L={L}$')
            elif with_error and name == 'q_ea':
                error = np.array([row['q2_sample_scatter_se']/(2*row['q_ea']) for row in rows])
                ax.errorbar(x, y, yerr=error, marker='o', capsize=3, lw=1.7,
                            label=fr'$L={L}$')
            else:
                ax.plot(x, y, marker='o', lw=1.7, label=fr'$L={L}$')
            ax.set_xlabel(r'$\Gamma/\sigma$')
            ax.set_ylabel(ylabel)
            ax.set_xticks([.5, 1, 2, 3, 5], labels=[r'$0.5$', r'$1$', r'$2$', r'$3$', r'$5$'])
            ax.grid(alpha=.22)
    axes[0].axhline(1, color='0.4', ls='--', lw=1, label=r'$\chi_{\mathrm{SG}}=1$')
    axes[0].legend(frameon=True)
    axes[0].set_title('Susceptibility')
    axes[1].set_title('RMS overlap')
    axes[2].set_title('Binder ratio')
    fig.suptitle(r'Gaussian bonds: exploratory screen ($n_{\mathrm{dis}}=2$; convergence unqualified)')
    fig.savefig(ROOT/'size_field_screen_latex.png', dpi=180)
    plt.close(fig)

    L = 6
    rows = [row for row in result['curves'] if row['L'] == L]
    ratio = np.array([row['gamma_over_sigma'] for row in rows])
    chi = np.array([row['chi_sg'] for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    sigmas = result['variance_scaling']['sigmas']
    colors = plt.get_cmap('viridis')(np.linspace(.08, .92, len(sigmas)))
    for sigma, color in zip(sigmas, colors):
        gamma = sigma*ratio
        ax.plot(gamma, chi, marker='o', lw=1.8, color=color,
                label=rf'$\mathrm{{Var}}(J)=\sigma^2={sigma**2:g}$')
    ax.set_xlabel(r'Transverse field, $\Gamma$')
    ax.set_ylabel(r'Spin-glass susceptibility, $\chi_{\mathrm{SG}}$')
    ax.set_title(rf'$L={L}$: exact Gaussian-bond variance rescaling')
    ax.grid(alpha=.22)
    ax.legend(title=r'Bond variance $\sigma^2$', frameon=True, loc='best')
    fig.savefig(ROOT/'variance_rescaling_latex.png', dpi=180)
    plt.close(fig)
    print('Rendered mathtext-labelled figures: size_field_screen_latex.png, variance_rescaling_latex.png')


if __name__ == '__main__':
    main()
