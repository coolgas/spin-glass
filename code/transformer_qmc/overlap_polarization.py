"""Decompose overlap with products across independent physical populations."""
from itertools import combinations

import numpy as np


def polarization_components(tags, counts):
    """Use equal-cycle descendant measures, as in projected P(q).

    Cross-population products avoid the positive self-product sampling bias.
    Their finite-sample estimates can be negative and must not be clipped.
    """
    tags = np.asarray(tags, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if tags.ndim != 4 or counts.shape != tags.shape[:-1]:
        raise ValueError('Expected tags[cycle,population,walker,site] and matching counts')
    if min(tags.shape) == 0 or tags.shape[1] < 2:
        raise ValueError('At least two populations and nonempty samples are required')
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isin(tags, [-1, 1]).all():
        raise ValueError('Expected finite nonnegative counts and Ising spins')
    walkers = tags.shape[2]
    np.testing.assert_allclose(counts.sum(axis=2), walkers, atol=1e-12, rtol=0)
    local = np.einsum('crw,crwn->crn', counts / walkers, tags).mean(axis=0)
    uniform = local.mean(axis=1)
    centered = local - uniform[:, None]
    pairs = list(combinations(range(tags.shape[1]), 2))
    q = float(np.mean([np.mean(local[a] * local[b]) for a, b in pairs]))
    uniform_squared = float(np.mean([uniform[a] * uniform[b] for a, b in pairs]))
    nonuniform = float(np.mean([np.mean(centered[a] * centered[b]) for a, b in pairs]))
    np.testing.assert_allclose(q, uniform_squared + nonuniform, atol=1e-12, rtol=0)
    return dict(mean_q=q, uniform_squared=uniform_squared, nonuniform=nonuniform,
                uniform_m=float(uniform.mean()), population_local_m=local.tolist())


def notebook_cells():
    import nbformat as nb

    return [nb.v4.new_markdown_cell(r'''## L=4, N=48: uniform polarization and nonuniform overlap

**The three lines in each panel** (brackets denote an equal average over disorders):

- **Black circles — mean overlap:** $[Q_J]_J=[N^{-1}\sum_i m_{i,J}^2]_J$.
- **Orange squares — uniform contribution:** $[U_J]_J=[\bar m_J^2]_J$, from the spatially averaged magnetization.
- **Blue triangles — nonuniform contribution:** $[V_J]_J=[N^{-1}\sum_i(m_{i,J}-\bar m_J)^2]_J$, from spatial variation of local magnetizations.

At each field, **black = orange + blue**. Error bars are $\pm1$ standard error across eight disorders.

For fixed disorder $J$, two independent configurations $s,t$ have overlap
$$q(s,t)=\frac1N\sum_i s_i t_i,\qquad
m_{i,J}=\langle Z_i\rangle_J,\qquad \bar m_J=\frac1N\sum_i m_{i,J}.$$
Independence gives $\langle s_i t_i\rangle_J=m_{i,J}^2$. Since
$\sum_i(m_{i,J}-\bar m_J)=0$, the mean overlap decomposes exactly as
$$Q_J\equiv\langle q\rangle_J=\frac1N\sum_i m_{i,J}^2
=\underbrace{\bar m_J^2}_{U_J:\ \text{uniform}}
+\underbrace{\frac1N\sum_i(m_{i,J}-\bar m_J)^2}_{V_J:\ \text{nonuniform}}.$$
The first term measures the contribution of **uniform polarization** to the
mean overlap. The second measures a **reproducible nonuniform local pattern**.
We evaluate both terms within each disorder, then average the eight disorders
equally: $[\bar m_J^2]_J$ is not $[\bar m_J]_J^2$.
This decomposes the **mean overlap**, not the histogram width or peak height.

The hypothesis is that the sharp, high-overlap low-$\Gamma$ peaks reflect
stronger freezing. A dominant nonuniform contribution would support the local
pattern part of that interpretation and show that uniform polarization explains
only a small fraction of the overlap. The inequality $h<\Gamma$ alone does not
exclude ferromagnetic order. Nonuniform polarization can also arise from a
field-pinned response to quenched disorder or other spatial order; it is not
by itself a glass order parameter in a field.

**Estimator from saved GFMC measurements.** Within one run and disorder (indices
suppressed), let $s_{cwi}^{(a)}$ be the tagged spin and $n_{cw}^{(a)}$ its
descendant count in population $a$, cycle $c$, walker $w$. With $C$ cycles,
$W$ walkers and $\sum_w n_{cw}^{(a)}=W$, compute
$$\widehat m_i^{(a)}=\frac1C\sum_{c=1}^C\sum_{w=1}^W
\frac{n_{cw}^{(a)}}W s_{cwi}^{(a)},\qquad
\widehat{\bar m}^{(a)}=\frac1N\sum_i\widehat m_i^{(a)},\qquad
\delta\widehat m_i^{(a)}=\widehat m_i^{(a)}-\widehat{\bar m}^{(a)}.$$
Average all six pairs of the four separately branching populations:
$$\widehat Q=\frac16\sum_{a<b}\frac1N\sum_i\widehat m_i^{(a)}\widehat m_i^{(b)},
\qquad \widehat U=\frac16\sum_{a<b}\widehat{\bar m}^{(a)}\widehat{\bar m}^{(b)},
\qquad \widehat V=\frac16\sum_{a<b}\frac1N\sum_i
\delta\widehat m_i^{(a)}\delta\widehat m_i^{(b)}.$$
Cross-population products avoid the positive self-squaring noise term
$\mathbb E[\widehat m^2]=(\mathbb E[\widehat m])^2+\operatorname{Var}(\widehat m)$;
they do not remove shared projection or mixing bias. The code checks
$$\widehat Q=\widehat U+\widehat V=\sum_q q\,\widehat P_J(q)$$
and agreement with the saved production mean overlap to tolerance $10^{-12}$.
Finite-sample estimates are not clipped to zero. This is distinct from the
single-snapshot second moment $\langle(N^{-1}\sum_i Z_i)^2\rangle$.

**Disorder averages and errors.** For any $X\in\{Q,U,V\}$, average two repeats
within each disorder, then $D=8$ disorders equally:
$$\widehat X_J=\frac{\widehat X_{J,0}+\widehat X_{J,1}}2,\qquad
\overline X=\frac1D\sum_J\widehat X_J,\qquad
\operatorname{SE}(\overline X)=
\sqrt{\frac{\sum_J(\widehat X_J-\overline X)^2}{D(D-1)}}.$$
The plotted curves are $\overline Q,\overline U,\overline V$ with capped
$\pm1$ disorder-SE bars. The nonuniform fraction is the ratio of disorder means,
$f=\overline V/\overline Q$. For its delete-one-disorder jackknife, define
$$f_{(-J)}=\frac{\sum_{K\ne J}\widehat V_K}{\sum_{K\ne J}\widehat Q_K},\qquad
\bar f_{\mathrm{leave}}=\frac1D\sum_J f_{(-J)},\qquad
\operatorname{SE}_{\mathrm{jack}}(f)=
\sqrt{\frac{D-1}{D}\sum_J(f_{(-J)}-\bar f_{\mathrm{leave}})^2}.$$
These errors describe variation across disorder estimates; they do not bound
systematic sampling or projection bias. Every ancestry alarm is retained with
equal weight.
'''), nb.v4.new_code_cell('''import importlib.util
polarization_source=root/'code/transformer_qmc/overlap_polarization.py'
spec=importlib.util.spec_from_file_location('overlap_polarization',polarization_source)
polarization_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(polarization_module)
polarization_runs=[]
for r in rows:
    t=r['task']
    assert t['L']==4 and t['audit']=='main' and r['replay_verified']
    raw_path=folder/r['raw_file']
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==r['raw_sha256']
    with np.load(raw_path) as raw:
        result=polarization_module.polarization_components(raw['tags'],raw['counts'])
    q=np.asarray(r['q_grid']);mass=np.asarray(r['mass'])
    np.testing.assert_allclose(result['mean_q'],q@mass,atol=1e-12,rtol=0)
    np.testing.assert_allclose(result['mean_q'],r['q'],atol=1e-12,rtol=0)
    polarization_runs.append(dict(task=t,raw_file=r['raw_file'],raw_sha256=r['raw_sha256'],
        provenance=r['provenance'],ancestry_alarm=r['ancestry_alarm'],**result))
polarization_disorders=[];polarization_summary=[]
component_keys=['mean_q','uniform_squared','nonuniform']
for gamma,kind in sorted({(r['task']['gamma'],r['task']['kind']) for r in polarization_runs}):
    group=[r for r in polarization_runs if (r['task']['gamma'],r['task']['kind'])==(gamma,kind)]
    assert len(group)==16
    values=[]
    for sid in range(60000,60008):
        repeats=[r for r in group if r['task']['sample_id']==sid]
        assert sorted(r['task']['repeat'] for r in repeats)==[0,1]
        value=np.mean([[r[key] for key in component_keys] for r in repeats],axis=0)
        values.append(value)
        polarization_disorders.append(dict(L=4,N=48,gamma=gamma,kind=kind,sample_id=sid,
            **dict(zip(component_keys,map(float,value))),
            ancestry_alarms=sum(r['ancestry_alarm'] for r in repeats)))
    values=np.asarray(values)
    np.testing.assert_allclose(values[:,0],values[:,1]+values[:,2],atol=1e-12,rtol=0)
    means=values.mean(axis=0);errors=values.std(axis=0,ddof=1)/np.sqrt(8)
    leave=np.asarray([np.delete(values,i,axis=0).mean(axis=0) for i in range(8)])
    assert means[0]>0 and (leave[:,0]>0).all(), 'Nonuniform fraction requires nonzero mean overlap'
    fractions=leave[:,2]/leave[:,0]
    fraction_se=float(np.sqrt(7/8*np.sum((fractions-fractions.mean())**2)))
    polarization_summary.append(dict(L=4,N=48,gamma=gamma,kind=kind,
        **dict(zip(component_keys,map(float,means))),
        **{key+'_se':float(error) for key,error in zip(component_keys,errors)},
        nonuniform_fraction=float(means[2]/means[0]),nonuniform_fraction_se=fraction_se,
        ancestry_alarms=sum(r['ancestry_alarm'] for r in group),disorders=8,runs=16))
fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
styles=[('mean_q','Mean overlap','black','o-'),
        ('uniform_squared','Uniform contribution','tab:orange','s--'),
        ('nonuniform','Nonuniform contribution','tab:blue','^-')]
for ax,kind in zip(axes,['jastrow','rbm_2']):
    data=[r for r in polarization_summary if r['kind']==kind]
    for key,label,color,fmt in styles:
        ax.errorbar([r['gamma'] for r in data],[r[key] for r in data],
                    yerr=[r[key+'_se'] for r in data],fmt=fmt,color=color,capsize=3,label=label)
    ax.set(title=names[kind]+' GFMC | L=4, N=48',xlabel='Gamma')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
axes[0].set_ylabel('Contribution to disorder mean overlap')
fig.suptitle('Overlap decomposition | periodic kagome | h=0.15 | eight disorders | ±1 disorder SE')
finish(fig,'l4_overlap_polarization')
headers=['Gamma','Guide','Mean q ± SE','Uniform contribution ± SE','Nonuniform contribution ± SE',
         'Nonuniform fraction ± jackknife SE','Ancestry alarms / 16']
table=[]
for r in polarization_summary:
    table.append([f"{r['gamma']:.2f}",names[r['kind']],
        *[f"{r[key]:.5f} ± {r[key+'_se']:.5f}" for key in component_keys],
        f"{r['nonuniform_fraction']:.3f} ± {r['nonuniform_fraction_se']:.3f}",r['ancestry_alarms']])
display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
    '\\n'.join('| '+' | '.join(map(str,r))+' |' for r in table)))
polarization_output=dict(h=.15,L=4,N=48,source_snapshot=str(folder.relative_to(root)),
    measurement_snapshot_utc=protocol.get('measurement_snapshot_utc'),
    source_sha256=hashlib.sha256(polarization_source.read_bytes()).hexdigest(),
    estimator='Cross-population products of equal-cycle descendant-weighted local magnetizations',
    uncertainty='Disorder SE; fraction uses delete-one-disorder jackknife; excludes mixing and projection bias',
    ground_state_qualified=False,runs=polarization_runs,disorders=polarization_disorders,summary=polarization_summary)
(figures/'l4_overlap_polarization.json').write_text(json.dumps(polarization_output,indent=2)+'\\n')
print(f'Checked decomposition against measured P(q) and production q for {len(polarization_runs)} runs.')
for kind in ['jastrow','rbm_2']:
    low=next(r for r in polarization_summary if r['kind']==kind and r['gamma']==.6)
    display(Markdown(f"At Gamma=0.60, **{names[kind]}**: the nonuniform contribution accounts for "
        f"**{100*low['nonuniform_fraction']:.1f}%** of the disorder mean overlap "
        f"(±{100*low['nonuniform_fraction_se']:.1f} percentage points, disorder jackknife SE)."))
'''), nb.v4.new_markdown_cell(r'''A dominant nonuniform contribution supports interpreting high overlap as a
reproducible spatial pattern rather than predominantly uniform polarization.
It does not establish spontaneous ferromagnetism or its absence, nor distinguish
glassy freezing from other nonuniform responses. That distinction continues to
rely on independent-start convergence, connected correlations, and size scaling.
The populations branch separately from the same frozen guide; agreement between
them does not by itself demonstrate exploration of competing sectors.
''')]
