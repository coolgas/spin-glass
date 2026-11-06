"""Collect validated equal-quota results and execute the planning notebook."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def collect(folder):
    archive={};rows=[];validation=[]
    for length in (6,7):
        for site in range(length):
            key=f'L{length}_site{site}'
            with h5py.File(folder/f'{key}.h5') as f:
                assert f.attrs['complete']
                np.testing.assert_array_equal(f['count'][:],np.full(20,100))
                offsets=f['positions_offset'][:]
                np.testing.assert_allclose(f['sigma_b'][:],np.sqrt(np.mean(np.diff(offsets,axis=-1)**2,axis=(1,2))),rtol=1e-14)
                archive[key+'_sigma_b']=f['sigma_b'][:]
                archive[key+'_mean_abs_displacement']=f['mean_abs_displacement'][:]
                archive['times']=f['times'][:];archive['c6_values']=f['c6_values'][:];archive['windows']=f['windows'][:]
                metrics=('mean_leakage','D','numerical_error','mean_leakage_integral','S','lambda_signed','lambda_absolute','resolved','S_ci','leakage_ci','lambda_signed_ci','lambda_absolute_ci')
                for metric in metrics:
                    archive[key+'_'+metric]=np.array([[f[f'c{ci}/b{m}/{metric}'][()] for m in range(20)] for ci in range(10)])
                for ci,c6 in enumerate(archive['c6_values']):
                    for m in range(20):
                        g=f[f'c{ci}/b{m}'];assert g.attrs['complete']
                        assert g['chi'].shape==(100,2000) and g['leakage'].shape==(100,2000)
                        signed=g['lambda_signed'][:];absolute=g['lambda_absolute'][:];resolved=g['resolved'][:]
                        np.testing.assert_allclose(absolute,abs(signed),equal_nan=True)
                        assert np.isfinite(absolute[resolved]).all() and np.isnan(absolute[~resolved]).all()
                        assert np.isfinite(g['D'][:]).all() and g['S'][()]>=0
                        row=dict(length=length,site=site,bin=m+1,count=100,c6=int(c6),sigma_b=float(f['sigma_b'][m]),
                                 mean_abs_displacement=float(f['mean_abs_displacement'][m]),S=float(g['S'][()]),mean_leakage=float(g['mean_leakage_integral'][()]))
                        for wi,(start,end) in enumerate(archive['windows']):
                            row[f'absolute_rate_{start:g}_{end:g}']=float(absolute[wi])
                            row[f'signed_rate_{start:g}_{end:g}']=float(signed[wi])
                        rows.append(row)
                validation.append(dict(ensemble=key,geometries=2000,complete_cells=200,resolved_primary_windows=int(archive[key+'_resolved'][:,:,:4].sum()),total_primary_windows=800))
    assert len(rows)==2600
    np.savez_compressed(folder/'summary.npz',**archive)
    with (folder/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'artifact_validation.json').write_text(json.dumps(dict(status='PASS',unique_geometries=26000,geometry_coefficient_evaluations=260000,ensemble_checks=validation),indent=2))


def execute_notebook(folder,template,destination):
    import nbformat as nbf
    from nbclient import NotebookClient
    nb=nbf.read(template,as_version=4)
    nb.cells=[c for c in nb.cells if not c.get('id','').startswith('computed-')]
    # Preserve the research framing and append clearly labeled executed results.
    nb.cells.append(nbf.v4.new_markdown_cell('## Executed leakage-sensitivity maps\n\nThe results below use all **100 saved geometries per site/bin**, with identical geometries across coefficients. The primary absolute-rate maps use equal-duration windows of 4, ending at **t=5,10,15,19.99**. Companion maps show leakage and displacement sensitivity at each endpoint. Gray cells denote unresolved logarithmic rates; they are not zero. All maps of the same quantity share color limits across sites and times. Full signed rates, bootstrap intervals and secondary windows are saved in the result archive.',id='computed-intro'))
    load="""from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, Markdown
relative=Path('results/leakage_sensitivity_20261102')
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/relative/'summary.npz').exists())
folder=root/relative
data=dict(np.load(folder/'summary.npz',allow_pickle=False))
report=json.loads((folder/'artifact_validation.json').read_text())
assert report['status']=='PASS' and report['unique_geometries']==26000
config=json.loads((folder/'config.json').read_text())
assert config['status']=='complete' and config['quota']==100
keys=[f'L{L}_site{k}' for L in (6,7) for k in range(L)]
snapshots=np.array([5.,10.,15.,19.99]);time_indices=np.rint(snapshots/.01).astype(int)
figdir=folder/'figures';figdir.mkdir(exist_ok=True)
display(Markdown('**Validated:** 26,000 distinct geometries; 260,000 geometry/coefficient evaluations. Every site/bin contains exactly 100 geometries. Input defect σ=1 µm; background σ=0.02 µm.'))
display(Markdown('**Exact C₆ grid:** '+', '.join(str(int(c)) for c in data['c6_values'])))
display(Markdown('**Numerically resolved primary rates:** '+str(sum(r['resolved_primary_windows'] for r in report['ensemble_checks']))+' / '+str(sum(r['total_primary_windows'] for r in report['ensemble_checks']))))
def bounds(values,log=False):
    v=np.concatenate([np.ravel(x) for x in values]);v=v[np.isfinite(v)]
    if len(v)==0:return Normalize(vmin=0,vmax=1)
    if log:
        v=v[v>0]
        return LogNorm(vmin=max(float(v.min()),1e-14),vmax=max(float(v.max()),1e-13))
    return Normalize(vmin=0,vmax=max(float(v.max()),1e-12))
norm_rate=bounds([data[k+'_lambda_absolute'][:,:,:4] for k in keys])
norm_leak=bounds([data[k+'_mean_leakage'][:,:,time_indices] for k in keys],True)
norm_d=bounds([data[k+'_D'][:,:,time_indices] for k in keys],True)
norm_s=bounds([data[k+'_S'] for k in keys],True)
norm_mean=bounds([data[k+'_mean_leakage_integral'] for k in keys],True)
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#bdbdbd')
def edges(x,log=False):
    v=np.log(x) if log else np.array(x)
    e=np.r_[v[0]-(v[1]-v[0])/2,(v[:-1]+v[1:])/2,v[-1]+(v[-1]-v[-2])/2]
    return np.exp(e) if log else e
def panel(ax,key,z,norm,title):
    x=data[key+'_sigma_b'];order=np.argsort(x)
    assert np.all(np.diff(x[order])>0)
    y=data['c6_values']
    shown=np.ma.masked_invalid(z[:,order])
    if isinstance(norm,LogNorm):shown=np.ma.masked_less_equal(shown,0)
    image=ax.pcolormesh(edges(x[order]),edges(y,True),shown,cmap=cmap,norm=norm,shading='flat')
    ax.set(yscale='log',xlabel='In-bin RMS bond deviation from 8 µm, σ_b (µm)',ylabel='C₆',title=title)
    ax.set_yticks(y);ax.set_yticklabels([f'{int(c):,}' for c in y],fontsize=6)
    ax.tick_params(axis='x',labelsize=8)
    return image
def save(fig,name,pdf=None):
    for ext in ('png','pdf'):fig.savefig(figdir/(name+'.'+ext),dpi=145,bbox_inches='tight')
    if pdf is not None:pdf.savefig(fig,bbox_inches='tight')
    plt.show();plt.close(fig)
"""
    nb.cells.append(nbf.v4.new_code_cell(load,id='computed-load'))
    nb.cells.append(nbf.v4.new_code_cell("""with PdfPages(figdir/'time_resolved_atlas.pdf') as atlas:
    for key in keys:
        fig,axes=plt.subplots(3,4,figsize=(19,12),constrained_layout=True)
        for col,t in enumerate(snapshots):
            a,b=data['windows'][col]
            specifications=[(data[key+'_lambda_absolute'][:,:,col],norm_rate,f'|λᶠᵀ| over [{a:g}, {b:g}]','Absolute rate (inverse time)'),
                            (data[key+'_mean_leakage'][:,:,time_indices[col]],norm_leak,f'Leakage at t={t:g}','Mean leakage'),
                            (data[key+'_D'][:,:,time_indices[col]],norm_d,f'Sensitivity at t={t:g}','RMS ∂ℓ/∂a (µm⁻¹)')]
            for row,(z,norm,title,label) in enumerate(specifications):
                image=panel(axes[row,col],key,z,norm,title)
                fig.colorbar(image,ax=axes[row,col],label=label,shrink=.8)
        fig.suptitle(f'{key}: equal-quota leakage response; 100 geometries/bin; input σ=1 µm\\nFixed nearest-neighbor blockade projector; identical C₆ grid and color limits across time')
        save(fig,key+'_time_resolved',atlas)
""",id='computed-time-maps'))
    nb.cells.append(nbf.v4.new_markdown_cell('### Integrated response and uncertainty\n\nThese panels use the full input window, t=0–19.99. Confidence intervals resample whole geometries within each site/bin; they describe finite-ensemble uncertainty. The absolute endpoint rate can be small despite large intermediate oscillations. It is not a chaos diagnostic, and a large relative rate at tiny sensitivity need not provide useful geometry information.',id='computed-integrated-intro'))
    nb.cells.append(nbf.v4.new_code_cell("""with PdfPages(figdir/'integrated_atlas.pdf') as atlas:
    for key in keys:
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for ax,metric,norm,title in zip(axes,('mean_leakage_integral','S'),(norm_mean,norm_s),('Time-averaged leakage','Time-integrated RMS sensitivity (µm⁻¹)')):
            image=panel(ax,key,data[key+'_'+metric],norm,title)
            fig.colorbar(image,ax=ax,shrink=.85)
        fig.suptitle(f'{key}; t=0–19.99; 100 geometries/bin; input σ=1 µm')
        save(fig,key+'_integrated',atlas)
""",id='computed-integrated'))
    nb.cells.append(nbf.v4.new_code_cell("""for L in (6,7):
    key=f'L{L}_site{L//2}'
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for col,ci in enumerate((0,5,9)):
        for m in (0,9,19):
            label=f'Bin {m+1}; σ_b={data[key+"_sigma_b"][m]:.3f} µm'
            axes[0,col].plot(data['times'],data[key+'_mean_leakage'][ci,m],label=label)
            axes[1,col].plot(data['times'][1:],data[key+'_D'][ci,m,1:],label=label)
        axes[0,col].set(title=f'C₆={int(data["c6_values"][ci]):,}',ylabel='Mean leakage')
        axes[1,col].set(yscale='log',xlabel='Time',ylabel='RMS leakage sensitivity (µm⁻¹)')
        for ax in axes[:,col]:ax.grid(alpha=.2)
        axes[0,col].legend(fontsize=7)
    fig.suptitle(f'{key}: time traces behind the maps; 100 geometries/bin')
    save(fig,key+'_time_traces')
""",id='computed-traces'))
    nb.cells.append(nbf.v4.new_markdown_cell('### Saved artifacts and next interpretation step\n\n`summary.csv` and `summary.npz` retain every coordinate, signed/absolute rate, numerical mask, and bootstrap interval. Per-site HDF5 files preserve all 100 source geometries, per-geometry leakage and derivatives (float32 storage), double-precision summaries, and accepted finite-difference steps. Computation uses float64/complex128.\n\nThese maps describe leakage response. Match them to the completed occupation-sensitivity and model-error study before interpreting them as an explanation of inference. Projected-Hamiltonian and observation-noise controls remain distinct follow-up experiments; they are not results of this scan.',id='computed-artifacts'))
    destination.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,destination)
    NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(destination.parent)}}).execute()
    nbf.validate(nb)
    nbf.write(nb,destination)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--notebook',type=Path,required=True)
    args=p.parse_args()
    collect(args.folder)
    execute_notebook(args.folder,args.template,args.notebook)
    (args.folder/'COMPLETE').touch()
