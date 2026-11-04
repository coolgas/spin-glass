"""Render completed pilot benchmarks, including exact small-system checks."""

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .experiment import load_model, write_json
from .vmc import enumerate_model


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    results = root/'data/kagome/transformer_af_dual_v1'
    output = results/'benchmark'
    output.mkdir(exist_ok=True)
    unified = json.loads((results/'unified_alibi/evaluation.json').read_text())
    seen = {r['L']:r for r in unified if r['sample_id']==0}
    fixed = {}
    for path in sorted((results/'fixed_size').glob('*/evaluation.json')):
        settings = json.loads((path.parent/'settings.json').read_text())
        if settings['gamma']==.5 and settings['h']==.3:
            fixed.setdefault(settings['L'],[]).append((settings,json.loads(path.read_text()),path.parent))
    model,sample,ham,settings = load_model(fixed[2][0][2])
    exact_fixed = enumerate_model(model,ham,.5,sample.points,2)
    exact = seen[2]['exact']
    enum_unified = seen[2]['enumerated_vmc']
    keys = ('energy','variance','q2','chi_connected','normalization')
    write_json(output/'small_system.json',dict(exact=exact,
               fixed={k:exact_fixed[k] for k in keys},unified=enum_unified))
    fig, axes = plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    sizes = [2,4,6,8]
    for index,L in enumerate(sizes):
        for j,(s,r,_) in enumerate(fixed[L]):
            axes[0].errorbar(index-.13+.06*j,r['energy']['mean']/(3*L*L),
                            yerr=r['energy']['se']/(3*L*L),fmt='o',color='tab:blue',
                            label='Fixed (both L6 seeds)' if index==0 and j==0 else None,capsize=3)
        r=seen[L]['vmc']['energy']
        axes[0].errorbar(index+.13,r['mean']/(3*L*L),yerr=r['se']/(3*L*L),
                        fmt='s',color='tab:orange',label='Unified ALiBi' if index==0 else None,capsize=3)
    axes[0].scatter([0],[exact['energy']/12],marker='*',s=120,color='black',label='Exact L2',zorder=5)
    axes[0].set(xticks=range(4),xticklabels=['2 (12)','4 (48)','6 (108)','8 (192)'],
                xlabel='L (number of spins)',ylabel='VMC energy per spin',title='Same disorder ID 0: lower is better')
    axes[0].legend(fontsize=8)
    names=['Exact','Fixed','Unified']
    values=[exact,exact_fixed,enum_unified]
    colors=['black','tab:blue','tab:orange']
    for ax,key,title in zip(axes[1:],['q2','chi_connected'],['L2 raw overlap ⟨q²⟩','L2 connected susceptibility']):
        bars=ax.bar(names,[v[key] for v in values],color=colors)
        ax.bar_label(bars,fmt='%.4f',padding=3)
        ax.set(title=title,ylim=(0,max(v[key] for v in values)*1.23))
    fig.suptitle('AutoDL pilot: Γ=0.5, h=0.3, positive random AF Kagome\n'
                 'Energy error bars: sampling SE; L2 observables: full enumeration. Different training budgets.',fontsize=11)
    fig.savefig(output/'fixed_vs_unified.png',dpi=180)
    fig.savefig(output/'fixed_vs_unified.pdf')
    print(json.dumps(dict(exact=exact,fixed={k:float(exact_fixed[k]) for k in keys},unified=enum_unified),indent=2))
    print(output/'fixed_vs_unified.png')


if __name__=='__main__':
    main()
