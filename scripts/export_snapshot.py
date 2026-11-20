"""Export a closed, matched-cohort histogram snapshot; preserve live runs."""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
run=args.run.expanduser().resolve()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=args.output.expanduser().resolve()
assert not out.is_relative_to(run), 'Export outside the source run'
out.mkdir(parents=True,exist_ok=False)
tasks=json.load(open(run/'tasks.json'))
groups={sid:[t for t in tasks if t['sample']==sid] for sid in range(128)}
ids=[sid for sid,tt in groups.items() if all((run/'cases'/t['id']/'result.json').exists() for t in tt)]
assert ids
models=['full','nn_nnn','nn'];deltas=[2.,3.15];drives=np.geomspace(.03,10.,30)
shape=(len(ids),3,2,30)
counts=np.zeros(shape+(49,),dtype=np.int64)
qea=np.zeros(shape);q2=np.zeros(shape);passed=np.zeros(shape,dtype=bool)
manifest=[];flags=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for si,sid in enumerate(ids):
 for t in groups[sid]:
  folder=run/'cases'/t['id'];rf=folder/'result.json';r=json.load(open(rf))
  pf=folder/'pq.npz';assert sha(pf)==r['hashes']['pq.npz']
  best=min(r['endpoints'],key=lambda e:e['energy'])
  k=(si,models.index(t['model']),deltas.index(t['delta']),int(np.argmin(abs(drives-t['omega']))))
  assert np.isclose(drives[k[-1]],t['omega'],rtol=1e-14)
  with np.load(pf) as z:
   assert np.array_equal(z['q'],np.linspace(-1,1,49))
   assert z['counts'].sum()==20000
   np.testing.assert_allclose(z['p'],z['counts']/20000,atol=1e-15)
   counts[k]=z['counts']
  qea[k]=best['observables']['qea'];q2[k]=best['observables']['q2']
  passed[k]=r['solver_pass'] and r['sampling_moment_check']
  manifest.append(dict(case=t['id'],result_sha256=sha(rf),pq_sha256=sha(pf),geometry_sha256=r['geometry_sha256']))
  if not passed[k]:flags.append(dict(case=t['id'],sample=sid,model=t['model'],delta=t['delta'],omega=t['omega'],solver_pass=r['solver_pass'],sampling_pass=r['sampling_moment_check'],start_comparison=r['start_comparison']))
assert (counts.sum(axis=-1)==20000).all()
np.savez_compressed(out/'histograms.npz',counts=counts,qea=qea,q2=q2,passed=passed,sample_ids=ids,models=models,deltas=deltas,drives=drives,q=np.linspace(-1,1,49))
metadata=dict(snapshot_utc=stamp,source_run=str(run),source_release='rydberg-kagome-20261117-v5',chi=12,D=128,sample_ids=ids,cohort_rule='All 180 cases completed; retain flagged cases, no passing-only selection',case_count=len(manifest),flags=flags,pairs_per_case=20000,files={'histograms.npz':sha(out/'histograms.npz')})
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
(out/'case-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=out/'snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in ['histograms.npz','metadata.json','case-manifest.json']:tar.add(out/name,arcname=name)
print(json.dumps(dict(directory=str(out),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,metadata=metadata),indent=2))
