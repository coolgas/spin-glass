from pathlib import Path
from types import SimpleNamespace
import json, pickle
from unittest.mock import patch
from transformer_qmc.triangle_correlation_dmrg import case, digest, write
import transformer_qmc.triangle_correlation_highchi as h
r=Path(__file__).parent
checks=r/'checks-v3'
checks.mkdir(exist_ok=True)
seen=[]
def defer(folder,model,mpo,psi,dimension,stage):
    seen.append(stage)
    if stage in ('seed0_D64','seed1_D64'): raise h.Deferred('intentional stage interruption')
try: case(checks,4,.5,60000,2,validate=True,m=4,resume=True,stage_guard=defer)
except h.Deferred: pass
else: raise AssertionError('deferral expected')
assert 'seed0_D32' in seen and 'seed1_D32' in seen, seen
resumed=[]
def observe(folder,model,mpo,psi,dimension,stage): resumed.append(stage)
row=case(checks,4,.5,60000,2,validate=True,m=4,resume=True,stage_guard=observe)
assert row['solver_passed'] and 'seed0_D32' not in resumed and 'seed1_D32' not in resumed
with (checks/'cases/L4_g0.5_s60000_m4_c2/model.pkl').open('rb') as f: model=pickle.load(f)['model']
assert all(w.nbytes == 4**4*2*8 for w in model['retained'])
# Existing worker records with no completed dependency must end visibly.
root=r/'controller-test'
root.mkdir(exist_ok=True)
write(root/'backend_benchmark.json',dict(validated=True,selected='cpu'))
job=dict(L=4,gamma=.5,sample_id=60000,m=4,chi=8,validate=True)
write(root/'attempts.json',{'case_'+h.key(job):dict(outcome='unresolved')})
with patch.object(h,'MS',[4]),patch.object(h,'CHIS',[8]),patch.object(h,'FIELDS',[.5]),patch.object(h,'IDS',[60000]):
    h.controller(SimpleNamespace(output=root,predecessor=[]))
status=json.loads((root/'status.json').read_text())
assert status['state']=='complete_with_unresolved_cases',status
assert len(status['blocked_dependencies'])==2,status
print('both-start deferral, hashed resume, compact retained basis, failed-dependency termination PASSED',flush=True)
# Test the actual CUDA hierarchy and its physical decoder against CPU.
from ising.hamiltonian import Hamiltonian
from transformer_qmc.physics import make_af_sample
from transformer_qmc.triangle_effective_rg import hierarchy
import numpy as np
sample=make_af_sample(4,60000)
ham=Hamiltonian(48,sample.bonds,sample.J,h=.15)
a=hierarchy(ham,.5,4,8)
b=hierarchy(ham,.5,4,8,eigensolver='cuda')
assert max(np.max(abs(x-y)) for x,y in zip(a['super_spectra'],b['super_spectra']))<1e-10
assert max(np.max(abs(x-y)) for x,y in zip(a['decoders'],b['decoders']))<1e-8
write(r/'controller_checks_passed.json',dict(passed=True))
print('actual CPU/CUDA hierarchy and decoders PASSED',flush=True)
