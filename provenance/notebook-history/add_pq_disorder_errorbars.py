from pathlib import Path
import os,base64,shutil
import nbformat
from nbclient import NotebookClient

path=Path('/Users/tianyiyan/Dropbox/spin-glasses/notebook/rydberg_kagome_L4_Pq_progress.ipynb')
stage=Path('/Users/tianyiyan/ResearchLocal/spin-glasses/staging/rydberg-L4-pq-errorbars-20261117')
stage.mkdir(parents=True,exist_ok=True)
backup=stage/'before-errorbars.ipynb'
if not backup.exists():shutil.copy2(path,backup)
nb=nbformat.read(path,as_version=4)
for cell in nb.cells:
    if cell.cell_type=='markdown':
        cell.source=cell.source.replace('\rangle',r'\rangle').replace('\x0crac',r'\frac')
nb.cells[4].source+='''

**Disorder error bars:** curves show the mean across the same 29 geometries. Shading and error bars show **± one sample standard deviation across geometries**, using ddof=1. Bars are drawn at every third overlap bin for readability; the band covers every bin. This is disorder spread, not the standard error or a 95% confidence interval for the mean. The observed histogram spread also includes finite Born-sampling noise. Negative lower error limits are statistical intervals, not negative probability masses; they are not clipped.
'''
nb.cells[5].source=nb.cells[5].source.replace(
    'fig, axes = plt.subplots(2, 4,',
    'sd_p = p.std(axis=0, ddof=1)\nfig, axes = plt.subplots(2, 4,')
nb.cells[5].source=nb.cells[5].source.replace(
    'ax.plot(q, mean_p[m,d,k], color=colors[m], lw=1.5, label=labels[m])',
    '''ax.errorbar(q, mean_p[m,d,k], yerr=sd_p[m,d,k], color=colors[m],
                        lw=1.3, elinewidth=.7, capsize=1.5, errorevery=3, label=labels[m])
            ax.fill_between(q, mean_p[m,d,k]-sd_p[m,d,k],
                            mean_p[m,d,k]+sd_p[m,d,k], color=colors[m], alpha=.10, lw=0)''')
nb.cells[5].source=nb.cells[5].source.replace('        ax.set_ylim(bottom=0)\n','')
new_md=nbformat.v4.new_markdown_cell(r'''## 3. Geometry-to-geometry spread of qEA

Error bars show **mean ± one sample standard deviation across the 29 geometries**, at every drive point. These quantify the spread between geometries, not uncertainty in the mean. The same geometry cohort is used for every curve, including the flagged case marked by a red cross. The following section retains the separate pointwise 95% bootstrap intervals for the disorder mean.

$$
\bar x=\frac1S\sum_{s=1}^{S}x_s,\qquad
s_x=\sqrt{\frac1{S-1}\sum_{s=1}^{S}(x_s-\bar x)^2},\qquad S=29.
$$

The standard error would be s_x/sqrt(S); it is not the quantity shown in these error bars. Neither measure includes RG systematic uncertainty.''')
new_code=nbformat.v4.new_code_cell('''sd_qea = qea.std(axis=0, ddof=1)
fig, axes = plt.subplots(1,2,figsize=(10,3.8),sharey=True,constrained_layout=True)
for d,ax in enumerate(axes):
    for m in range(3):
        ax.errorbar(drives, mean_qea[m,d], yerr=sd_qea[m,d],
                    color=colors[m], label=labels[m], lw=1.3,
                    elinewidth=.8, capsize=2, marker='.', markersize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Drive Ω/V₀')
    ax.text(.03,.04,f'Δ/V₀={deltas[d]:g}',transform=ax.transAxes)
axes[0].set_ylabel('Mean qEA ± disorder SD')
axes[0].plot(10,mean_qea[2,0,-1],'x',color='red',ms=8,mew=2)
axes[1].legend(frameon=False)
plt.show()
''')
nb.cells[6].source=nb.cells[6].source.replace('## 3. Mean overlap and disorder uncertainty','## 4. Uncertainty in the disorder mean (95% bootstrap intervals)')
nb.cells[6:6]=[new_md,new_code]
os.environ['MPLCONFIGDIR']=str(stage/'mpl-cache')
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(stage)}}).execute()
nbformat.validate(nb)
for c in nb.cells:
    assert not any(ord(ch)<32 and ch not in '\n\t' for ch in c.source)
nbformat.write(nb,path)
nbformat.write(nb,stage/path.name)
i=0
for c in nb.cells:
    for output in c.get('outputs',[]):
        assert output.output_type!='error'
        if 'image/png' in output.get('data',{}):
            i+=1;(stage/f'plot-{i}.png').write_bytes(base64.b64decode(output['data']['image/png']))
print('Updated and executed:',path,'figures:',i)
