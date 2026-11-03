"""Execute a measured higher-field P(q) report on AutoDL."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import nbformat as nb
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .production_protocol import identity


def freeze_preview(source, folder):
    """Capture only complete measured groups without racing the active writer."""
    rows = json.loads((source/'projected_distributions.json').read_text())['rows']
    protocol = json.loads((source/'protocol.json').read_text())
    complete = []
    for gamma in protocol['fields']:
        for kind in ['jastrow', 'rbm_2']:
            group = [r for r in rows if (r['task']['gamma'], r['task']['kind']) == (gamma, kind)]
            if len(group) == 16 and all(sorted(r['task']['repeat'] for r in group
                    if r['task']['sample_id'] == sid) == [0, 1] for sid in range(60000, 60008)):
                complete.extend(group)
    folder.mkdir(parents=True, exist_ok=False)
    for name in protocol['input_sha256']:
        dest = folder/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, dest)
    for row in complete:
        dest = folder/row['raw_file']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/row['raw_file'], dest)
        key = identity(row['task'])[:24]
        checkpoint = Path('projected_histograms/checkpoints')/(key+'.npz')
        if (source/checkpoint).exists():
            dest = folder/checkpoint
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/checkpoint, dest)
            protocol['input_sha256'][str(checkpoint)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    # Include dependencies so the exported notebook can rerun from its snapshot.
    code_root = Path(__file__).parent.parent
    sources = list(code_root.glob('*.py'))
    for package in ['transformer_qmc', 'gfmc', 'ising', 'kagome_qea']:
        sources.extend((code_root/package).rglob('*.py'))
    for path in sources:
        relative = Path('sources')/path.relative_to(code_root)
        dest = folder/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        protocol['input_sha256'][str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    protocol['measurement_snapshot_utc'] = datetime.now(timezone.utc).isoformat()
    protocol['active_histogram_campaign'] = str(source)
    protocol['measured_coverage'] = [dict(gamma=gamma, kind=kind,
        runs=sum((r['task']['gamma'], r['task']['kind']) == (gamma, kind) for r in complete))
        for gamma in protocol['fields'] for kind in ['jastrow', 'rbm_2']]
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (folder/'projected_distributions.json').write_text(json.dumps(dict(rows=complete), indent=2)+'\n')
    (folder/'complete.json').write_text(json.dumps(dict(projected=len(complete), failures=[],
        scope='Complete measured ensembles at preview capture; other fields remain pending.'), indent=2)+'\n')


def build(folder, root):
    relative = folder.relative_to(root)
    cells = [nb.v4.new_markdown_cell(r'''# Broader overlap distributions at fixed $h=0.15$

These are measured pure GFMC overlap histograms for periodic kagome $L=4$
($N=48$). Only complete ensembles with eight disorders and two repeats per
disorder are shown. The same disorder IDs 60000–60007 have equal weight at every
field and for both guides; no disorder is selected by its observed broadness.
The earlier [exact $L=2$ and guide-MCMC comparisons](preliminary_disorder_overlap.html)
remain available in the companion report.

The Hamiltonian is
$$H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j-0.15\sum_iZ_i-\Gamma\sum_iX_i,$$
with iid antiferromagnetic bonds $K_{ij}\sim U(0.5,1.5)$, stored as $J_{ij}=-K_{ij}$.
NumPy `default_rng(SeedSequence([20261027,L,sample_id,1]))` draws bonds in the
periodic lattice bond-list order. Quenched bonds are reused across all fields,
guides, and seeds for each size and disorder ID.

Each run uses four independently branching populations with 512 walkers each,
equilibration and forward times 16, and eight cycles. Descendant-weighted
configurations are pooled equally across cycles; all six cross-population pairs
give $q=N^{-1}\sum_i s_i t_i$. Repeats are averaged within each disorder first.
The higher-field histograms are deterministic replays of frozen production
trajectories. Their $q,q^2$, and connected spectra must match production to
$10^{-12}$. The $\Gamma=0.60,0.65$ histograms reuse the earlier verified measurements.

$P(q)$ denotes probability mass at each allowed overlap; connecting lines are
visual aids. Capped bars give $\pm s_J/\sqrt8$ pointwise in $q$. Width bars use a
delete-one-disorder jackknife. These errors exclude mixing and projection bias.
All ancestry alarms retain equal disorder weight. **The projections remain
preliminary** pending population, forward-time and starting-state sensitivity.
Broadening alone does not establish glassiness, a transition, or finite-size drift.
Computation, checks, plotting and notebook execution take place on AutoDL.''')]
    cells.append(nb.v4.new_code_cell(f'''%matplotlib inline
from pathlib import Path
import hashlib, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/kagome').is_dir())
folder=root/{str(relative)!r}
import sys
sys.path.insert(0,str(folder/'sources'))
from transformer_qmc.production_protocol import identity
from transformer_qmc.disorder_scaling import observables
load=lambda name:json.loads((folder/name).read_text())
protocol=load('protocol.json')
assert protocol['h']==.15 and not load('complete.json')['failures']
for name,expected in protocol['input_sha256'].items():
    assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
rows=load('projected_distributions.json')['rows']
assert len(rows)==load('complete.json')['projected']
groups=defaultdict(list)
for r in rows:
    t=r['task']; assert t['L']==4 and t['audit']=='main' and r['replay_verified']
    q=np.array(r['q_grid']);p=np.array(r['mass'])
    assert (p>=0).all()
    np.testing.assert_allclose(p.sum(),1,atol=1e-12)
    np.testing.assert_allclose([q@p,q**2@p],[r['q'],r['q2']],atol=1e-12,rtol=0)
    original=load('production/runs/'+identity(t)[:24]+'.json')
    np.testing.assert_allclose([r['q'],r['q2']],[original['q'],original['q2']],atol=1e-12,rtol=0)
    np.testing.assert_allclose(r['spectrum'],original['spectrum'],atol=1e-12,rtol=0)
    assert r['provenance']['model_sha256']==original['provenance']['model_sha256']
    assert hashlib.sha256((folder/r['raw_file']).read_bytes()).hexdigest()==r['raw_sha256']
    with np.load(folder/r['raw_file']) as raw:
        weights=raw['counts']/t['walkers']
        r['uniform_m']=float((weights*raw['tags'].mean(3)).sum(2).mean())
        spins=raw['tags'].astype(float)
        m=np.einsum('crw,crwn->crn',weights,spins)
        pair=np.einsum('crw,crwi,crwj->crij',weights,spins,spins)
        reconstructed=observables(m.mean(0),pair.mean(0),t['L'])
        np.testing.assert_allclose([q@p,q**2@p],[reconstructed['q'],reconstructed['q2']],atol=1e-12,rtol=0)
        np.testing.assert_allclose(r['spectrum'],reconstructed['spectrum'],atol=1e-12,rtol=0)
    groups[(t['gamma'],t['kind'])].append(r)
def width(a):
    mean=a.mean(0)
    return np.sqrt(max(0,float(q*q@mean-(q@mean)**2)))
def jackknife(a,stat):
    x=np.array([stat(np.delete(a,i,axis=0)) for i in range(8)])
    return float(np.sqrt(7/8*np.sum((x-x.mean())**2)))
curves={{}};summary=[]
for (g,kind),rs in sorted(groups.items()):
    assert len(rs)==16
    byid={{sid:[r for r in rs if r['task']['sample_id']==sid] for sid in range(60000,60008)}}
    assert all(sorted(r['task']['repeat'] for r in sample)==[0,1] for sample in byid.values())
    values=np.array([np.mean([r['mass'] for r in sample],axis=0) for sample in byid.values()])
    magnet=np.array([np.mean([r['uniform_m'] for r in sample]) for sample in byid.values()])
    q=np.array(rs[0]['q_grid']);mean=values.mean(0);se=values.std(0,ddof=1)/np.sqrt(8)
    assert all(np.array_equal(q,r['q_grid']) for r in rs)
    curves[(g,kind)]=(q,values,mean,se)
    m=values@q;v=values@(q*q)-m*m
    within=float(v.mean());between=float(m.var())
    np.testing.assert_allclose(within+between,width(values)**2,atol=1e-12)
    negative=values[:,q<0].sum(1)
    summary.append(dict(L=4,gamma=g,kind=kind,width=float(width(values)),width_se=jackknife(values,width),
        mean_q=float(m.mean()),mean_q_se=float(m.std(ddof=1)/np.sqrt(8)),peak_q=float(q[mean.argmax()]),
        negative_mass=float(negative.mean()),negative_mass_se=float(negative.std(ddof=1)/np.sqrt(8)),
        uniform_m=float(magnet.mean()),uniform_m_se=float(magnet.std(ddof=1)/np.sqrt(8)),
        within_disorder_variance=within,between_disorder_variance=between,
        ancestry_alarms=sum(r['ancestry_alarm'] for r in rs),runs=16))
figures=folder/'figures';figures.mkdir(exist_ok=True)
def finish(fig,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(figures/(name+'.'+suffix),dpi=180,bbox_inches='tight')
    plt.show()
names={{'jastrow':'Jastrow','rbm_2':'RBM (alpha=2)'}}
colors={{.6:'#440154',.65:'#3b528b',.7:'#21918c',.85:'#e08b20',1.:'#cc3311'}}
print('Frozen inputs:',protocol['created_utc'])
print('Measurement capture:',protocol.get('measurement_snapshot_utc',protocol['created_utc']))
print('Verified histogram runs:',len(rows),'| h=0.15 | eight equally weighted disorders per curve')
for item in protocol['coverage']:
    if (item['gamma'],item['kind']) not in curves:print('Histogram ensemble pending:',item)
'''))
    cells.append(nb.v4.new_code_cell('''fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
for ax,kind in zip(axes,['jastrow','rbm_2']):
    for (g,k),(q,values,mean,se) in sorted(curves.items()):
        if k!=kind:continue
        ax.errorbar(q,mean,yerr=se,fmt='o-',ms=2.8,lw=1.1,elinewidth=.7,capsize=2,
                    capthick=.7,color=colors[g],label=f'Gamma={g:.2f}')
    ax.set(title=names[kind]+' guide: after projection',xlabel='Overlap q',xlim=(-1.02,1.02))
    pending=[g for g in protocol['fields'] if (g,kind) not in curves]
    if pending:
        ax.text(.02,.03,'Pending: '+', '.join(f'{g:.2f}' for g in pending),transform=ax.transAxes,fontsize=8)
    ax.grid(alpha=.18);ax.legend(fontsize=9)
axes[0].set_ylabel('Disorder mean P(q), probability mass')
fig.suptitle('Measured L=4 overlap distributions | h=0.15 | eight disorders | bars: ±1 disorder SE')
finish(fig,'broader_distributions')
'''))
    cells.append(nb.v4.new_markdown_cell(r'''The following diagnostics use the same
histograms and raw descendant measurements. The width is that of the
disorder-averaged distribution. Its variance splits into the average
within-disorder variance and the variance of disorder means; this distinguishes
changes within samples from a spread of sample centers. Uniform magnetization
and negative-overlap mass are displayed alongside width because the fixed
longitudinal field breaks spin-inversion symmetry.'''))
    cells.append(nb.v4.new_code_cell('''fig,axes=plt.subplots(2,2,figsize=(11,8))
for ax,key,error,label in zip(axes.flat,['width','mean_q','negative_mass','uniform_m'],
        ['width_se','mean_q_se','negative_mass_se','uniform_m_se'],
        ['Width sigma(q)','Mean overlap','P(q < 0)','Uniform magnetization']):
    for kind,color in [('jastrow','tab:blue'),('rbm_2','tab:orange')]:
        data=[r for r in summary if r['kind']==kind]
        ax.errorbar([r['gamma'] for r in data],[r[key] for r in data],yerr=[r[error] for r in data],
                    fmt='o-',capsize=3,color=color,label=names[kind])
    ax.set(xlabel='Gamma',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('L=4, fixed h=0.15 | same eight disorders | ±1 disorder SE')
finish(fig,'broader_distribution_diagnostics')
headers=['Gamma','Guide','Width ± SE','Mean q','P(q<0)','Uniform m','Within variance','Between variance','Ancestry alarms / 16']
table=[]
for r in summary:
    table.append([f"{r['gamma']:.2f}",names[r['kind']],f"{r['width']:.4f} ± {r['width_se']:.4f}",
                  f"{r['mean_q']:.4f}",f"{r['negative_mass']:.5f}",f"{r['uniform_m']:.4f}",
                  f"{r['within_disorder_variance']:.5f}",f"{r['between_disorder_variance']:.5f}",r['ancestry_alarms']])
display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
    '\\n'.join('| '+' | '.join(map(str,r))+' |' for r in table)))
(folder/'summary.json').write_text(json.dumps(dict(created_utc=protocol['created_utc'],
    measurement_snapshot_utc=protocol.get('measurement_snapshot_utc'),
    complete_ensembles=summary,production_coverage=protocol['coverage'],
    measured_coverage=protocol.get('measured_coverage'),ground_state_qualified=False),indent=2)+'\\n')
print('Verified frozen input hashes, raw measurement hashes, replayed moments and spectra, and variance decomposition.')
'''))
    cells.append(nb.v4.new_code_cell('''# Paired field comparisons use the same eight disorders in the same order.
paired=[]
for kind in ['jastrow','rbm_2']:
    base=curves[(.6,kind)][1]
    for (g,k),(q,values,mean,se) in sorted(curves.items()):
        if k!=kind or g<=.6:continue
        change=width(values)-width(base)
        leave=np.array([width(np.delete(values,i,axis=0))-width(np.delete(base,i,axis=0)) for i in range(8)])
        error=float(np.sqrt(7/8*np.sum((leave-leave.mean())**2)))
        within=lambda a:a@(q*q)-(a@q)**2
        delta=within(values)-within(base)
        paired.append(dict(kind=kind,gamma=g,reference_gamma=.6,width_change=float(change),
            paired_width_change_se=error,width_ratio=float(width(values)/width(base)),
            within_variance_change=float(delta.mean()),within_variance_change_se=float(delta.std(ddof=1)/np.sqrt(8))))
headers=['Guide','Gamma','Width change from 0.60 ± paired SE','Width ratio','Within-disorder variance change ± SE']
display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
    '\\n'.join('| '+' | '.join([names[r['kind']],f"{r['gamma']:.2f}",
        f"{r['width_change']:.4f} ± {r['paired_width_change_se']:.4f}",f"{r['width_ratio']:.3f}",
        f"{r['within_variance_change']:.5f} ± {r['within_variance_change_se']:.5f}"])+' |' for r in paired)))
saved=json.loads((folder/'summary.json').read_text());saved['paired_field_comparisons']=paired
(folder/'summary.json').write_text(json.dumps(saved,indent=2)+'\\n')
'''))
    cells.append(nb.v4.new_markdown_cell(r'''The fine scan retains the full
$L=4$ grid $0.55,0.56,\ldots,0.70$. Original $L=6$ coverage remains
$0.60,0.65,0.70,0.85,1.00$, supplemented by the lower-field fine grid.
No incomplete ensemble is plotted here. Frozen production measurements, guide
models for new replays, source files, seeds, checkpoint ancestry and failures
remain with this snapshot or the source campaigns; replay checkpoints are retained.
No matched-size crossing or collapse is inferred from this report.'''))
    notebook = nb.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    output = root/'notebook/broader_overlap_distributions.ipynb'
    output.parent.mkdir(parents=True, exist_ok=True)
    nb.write(notebook, output)
    NotebookClient(notebook, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    nb.write(notebook, output)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    output.with_suffix('.html').write_text(html)
    files = [p for p in folder.rglob('*') if p.is_file()]
    files.extend([output, output.with_suffix('.html')])
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)}
    (root/'broader_overlap_delivery_sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Executed:', output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--preview', type=Path, help='Freeze complete measured groups here before executing.')
    a = p.parse_args()
    if a.preview:
        freeze_preview(a.folder, a.preview)
        a.folder = a.preview
    build(a.folder, a.root)
