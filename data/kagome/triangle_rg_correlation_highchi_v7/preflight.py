from pathlib import Path
import json, numpy as np
from transformer_qmc.triangle_correlation_dmrg import case, write
from transformer_qmc.triangle_correlation_highchi import backend_benchmark, estimates, guard, Deferred, worker_limits
root=Path(__file__).parent
checks=root/'checks'
checks.mkdir(exist_ok=True)
calls=[]
def interrupt(folder, model, mpo, psi, dimension, stage):
    calls.append(stage)
    if stage=='seed0_D64': raise Deferred('intentional interruption to test checkpoint recovery')
try:
    case(checks,4,.5,60000,2,validate=True,m=4,resume=True,stage_guard=interrupt)
except Deferred: pass
assert (checks/'cases/L4_g0.5_s60000_m4_c2/seed0_D32.pkl').exists()
continued=[]
def observe(folder,model,mpo,psi,dimension,stage): continued.append(stage)
r=case(checks,4,.5,60000,2,validate=True,m=4,resume=True,stage_guard=observe)
assert r['solver_passed'], r
assert 'seed0_D32' not in continued and 'exact' not in continued, continued
previous=Path('/root/autodl-tmp/triangle-correlation-retention-20261105-v6/results/cases/L4_g0.5_s60000_m4_c2/result.json')
a=json.loads(previous.read_text())
assert abs(a['records'][-1]['energy']-r['records'][-1]['energy'])/48 < 1e-10
assert np.max(abs(np.array(a['records'][-1]['spectra'])-r['records'][-1]['spectra']))<1e-8
print('checkpoint recovery and unchanged low-chi validation passed',flush=True)
r8=case(checks,4,.5,60000,8,validate=True,m=4,resume=True,stage_guard=guard)
assert r8['solver_passed'], r8
print('chi8 exact reduced-space, MPO, physical moments and two starts passed',flush=True)
import pickle
with (checks/'cases/L4_g0.5_s60000_m4_c8/model.pkl').open('rb') as f: saved=pickle.load(f)
model,mpo=saved['model'],saved['mpo']
assert estimates(model,mpo,512)['memory_gib'] < 3
import os
os.environ['HIGHCHI_MEMORY_GIB']='0.01'
try: guard(checks,model,mpo,None,512,'intentional_resource_limit')
except Deferred: pass
else: raise AssertionError('stage guard did not defer')
assert json.loads((checks/'deferred.json').read_text())['reason']=='next stage exceeds this worker memory reservation'
assert worker_limits({'production_estimates':{'32':{'memory_gib':200,'checkpoint_gib':1}}},100) is None
os.environ.pop('HIGHCHI_MEMORY_GIB')
backend_benchmark(root/'results')
write(root/'preflight_passed.json',dict(passed=True,low_chi_unchanged=True,resume_verified=True,chi8_validation=True,resource_deferral=True))
print('ALL PREFLIGHT CHECKS PASSED',flush=True)
