from pathlib import Path
import base64,json,os,datetime,hashlib
import nbformat
from nbclient import NotebookClient

stage=Path('/Users/tianyiyan/ResearchLocal/spin-glasses/staging/rydberg-L4-pq-preview-20261118T004328Z')
path=Path('/Users/tianyiyan/Dropbox/spin-glasses/notebook/rydberg_kagome_L4_Pq_progress.ipynb')
nb=nbformat.read(path,as_version=4)
meta=json.load(open(stage/'metadata.json'))
n=len(meta['sample_ids'])
for cell in nb.cells:
    if cell.cell_type=='markdown':
        cell.source=cell.source.replace('29 geometries',f'{n} geometries').replace('S=29',f'S={n}')
nb.cells[0].source=nb.cells[0].source.replace('17 November 2026, 12:41 UTC','18 November 2026, 00:43 UTC').replace('29 completed geometries (IDs 0–28)',f'{n} completed geometries (IDs 0–71)').replace('5,220 cases','12,960 cases')
start=nb.cells[0].source.index('**One flagged case')
end=nb.cells[0].source.index('\n\nAll plotting data',start)
nb.cells[0].source=nb.cells[0].source[:start]+'''**Two flagged cases are included:** geometry 25, NN, Delta/V0=2, drive=10; and geometry 38, NN, Delta/V0=3.15, drive≈8.1847. In each case one start failed the sweep-stability criterion at 120 sweeps; final energies and qEA agree closely between starts and sampling passes. Both affected points are marked below. Keeping these cases avoids a passing-only disorder selection.'''+nb.cells[0].source[end:]
c=nb.cells[1].source
head=c[:c.index('METADATA =')]
tail=c[c.index('raw ='):].replace('(29, 3, 2, 30, 49)','(len(sample_ids), 3, 2, 30, 49)').replace('flagged case retained.','flagged cases retained.')
nb.cells[1].source=head+'METADATA = json.loads('+repr(json.dumps(meta))+')\nPAYLOAD = '+repr(base64.b64encode((stage/'histograms.npz').read_bytes()).decode())+'\n'+tail+'''
flag_points = []
for flag in METADATA['flags']:
    m = list(arrays['models']).index(flag['model'])
    d = int(np.argmin(abs(deltas-flag['delta'])))
    k = int(np.argmin(abs(drives-flag['omega'])))
    flag_points.append((m,d,k,flag['sample']))
assert (~passed).sum() == len(METADATA['flags'])
'''
nb.cells[2].source=nb.cells[2].source.replace('The red cross identifies the drive containing the flagged case.','Red crosses identify model/detuning/drive points containing flagged cases.')
nb.cells[3].source=nb.cells[3].source.replace("axes[0,2].plot(10, mean_qea[2,0,-1], 'x', color='red', ms=8, mew=2)","for m,d,k,sid in flag_points:\n    axes[d,m].plot(drives[k], mean_qea[m,d,k], 'x', color='red', ms=8, mew=2)")
nb.cells[4].source=nb.cells[4].source.replace('The high-drive NN curve at detuning 2 includes the flagged case.','The high-drive NN curve at detuning 2 includes one flagged case; the other flagged drive (8.1847, detuning 3.15) is not among these selected slices.')
nb.cells[6].source=nb.cells[6].source.replace('including the flagged case marked by a red cross','including the flagged cases marked by red crosses')
nb.cells[8].source=nb.cells[8].source.replace('The red cross marks the affected high-drive point.','Red crosses mark the two affected high-drive points.')
for idx in [7,9]:
    nb.cells[idx].source=nb.cells[idx].source.replace("axes[0].plot(10,mean_qea[2,0,-1],'x',color='red',ms=8,mew=2)","for m,d,k,sid in flag_points:\n    axes[d].plot(drives[k],mean_qea[m,d,k],'x',color='red',ms=8,mew=2)")
nb.cells[9].source=nb.cells[9].source.replace('bootstrap = qea[resamples].mean(axis=1)',"# Bounded batches avoid materializing all resampled curves at once.\nbootstrap = np.concatenate([qea[batch].mean(axis=1)\n                            for batch in np.array_split(resamples,20)], axis=0)")
nb.cells[9].source=nb.cells[9].source.replace('sensitivity to the single flagged geometry','sensitivity to flagged geometries')
nb.cells[9].source=nb.cells[9].source[:nb.cells[9].source.index('keep = sample_ids != 25')]+'''
for m,d,k,sid in flag_points:
    reduced = p[sample_ids != sid,m,d,k].mean(axis=0)
    shift = np.max(abs(np.cumsum(mean_p[m,d,k])-np.cumsum(reduced)))
    print(f'{labels[m]}, Δ={deltas[d]:g}, drive={drives[k]:.5g}: omitting geometry {sid} changes mean CDF by {shift:.6f}.')
print('These omission sensitivities change the cohort; they are not corrections or bounds on solver error.')
'''
nb.cells[10].source=nb.cells[10].source.replace('660113ea84c5354b972160b752beaef035a90fb8805604c50af724f332c0d675','86b01a45ffee89e85ebdfb316c36ccf4b511c787a7bcf3d8863dc7076f1ccc72').replace('rydberg-L4-pq-preview-20261117T124125Z','rydberg-L4-pq-preview-20261118T004328Z')
nb.metadata['snapshot_utc']=meta['snapshot_utc']
os.environ['MPLCONFIGDIR']=str(stage/'mpl-cache')
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(stage)}}).execute()
nbformat.validate(nb)
for cell in nb.cells:
    assert not any(ord(ch)<32 and ch not in '\n\t' for ch in cell.source)
    if cell.cell_type=='markdown':
        assert '29 ' not in cell.source
        assert cell.source.count('$$')%2==0
nbformat.write(nb,stage/path.name)
nbformat.write(nb,path)
i=0
for cell in nb.cells:
    for output in cell.get('outputs',[]):
        assert output.output_type!='error'
        if 'image/png' in output.get('data',{}):
            i+=1;(stage/f'plot-{i}.png').write_bytes(base64.b64decode(output['data']['image/png']))
        if output.output_type=='stream':print(output.text)
record=dict(notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),snapshot_utc=meta['snapshot_utc'],geometries=n,cases=meta['case_count'],figures=i,executed=True)
(stage/'notebook-publication.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
