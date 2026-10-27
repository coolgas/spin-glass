import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import kagome_qea.pqmc as pq
from kagome_qea.sample import make_sample
source=Path(pq.__file__).read_text().splitlines()
end_line=next(i for i,s in enumerate(source,1) if s.strip().startswith('tags = np.fromiter((walker.tag for walker in saved)'))
start_line=next(i for i,s in enumerate(source,1) if s.strip()=='for identity, walker in enumerate(walkers):')
rows=[]
def local_trace(frame,event,arg):
    if event!='line':return local_trace
    f=frame.f_locals
    if frame.f_lineno==start_line and rows and '_saved_weights' in rows[-1]:
        weights=np.array([w.weight for w in f['walkers']])
        if rows:
            rows[-1]['next_cycle_weights_equal_saved']=bool(np.array_equal(weights,rows[-1].pop('_saved_weights')))
            rows[-1]['next_cycle_nonunit_weights']=int(np.count_nonzero(weights!=1.))
    if frame.f_lineno==end_line and f['rejected'] is not None:
        a,b=f['saved'],f['backup']
        rows.append(dict(cycle=len(rows),achieved=float(f['elapsed']),rejected_dt=float(f['rejected'][0]),
            changed_spins=sum(not np.array_equal(x.spins,y.spins) for x,y in zip(a,b)),
            changed_diagonals=sum(x.diagonal!=y.diagonal for x,y in zip(a,b)),
            changed_weights=sum(x.weight!=y.weight for x,y in zip(a,b)),
            saved_weight_min=float(min(x.weight for x in a)),saved_weight_max=float(max(x.weight for x in a)),
            backup_weight_min=float(min(x.weight for x in b)),backup_weight_max=float(max(x.weight for x in b)),
            _saved_weights=[x.weight for x in a]))
    return local_trace
def global_trace(frame,event,arg):
    return local_trace if frame.f_code is pq._project_once.__code__ else None
start=time.monotonic()
sys.settrace(global_trace)
try:
    r=pq._project_once(make_sample(6,True,0).hamiltonian(0.),.05,
        np.random.default_rng(np.random.SeedSequence(20261025).spawn(32)[0]),
        n_walkers=5000,tau_block=.2,n_equilibrate=40,n_measure=40,tau_forward=8.,n_forward_cycles=8)
finally:sys.settrace(None)
for row in rows:row.pop('_saved_weights',None)
with np.load(ROOT/'runs/fresh_L6_g005_t8_p0.npz') as baseline:
    assert float(baseline['chi_sg'])==r['chi_sg']
    assert np.array_equal(baseline['forward_times'],r['forward_times'])
output=dict(case='L6 sample0 gamma=.05 seed20261025 child0 walkers5000',
    chi=float(r['chi_sg']),times=r['forward_times'].tolist(),rows=rows,seconds=time.monotonic()-start)
(ROOT/'kagome_rollback.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
