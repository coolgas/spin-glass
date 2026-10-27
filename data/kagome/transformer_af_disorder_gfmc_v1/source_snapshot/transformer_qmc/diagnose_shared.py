"""Enumerated frozen-checkpoint symmetry and disorder diagnostics at 12 spins."""

import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import eigsh

from kagome_qea.exact import operator_matrix
from .experiment import load_checkpoint, write_json
from .physics import basis, correlations, make_af_sample
from .unified import UnifiedTransformer


def ground(matrix):
    values, vectors = eigsh(matrix, k=1, which='SA', tol=1e-12,
                            v0=np.random.default_rng(731).normal(size=matrix.shape[0]))
    psi = vectors[:, 0]
    if psi.sum() < 0:
        psi = -psi
    assert np.linalg.norm(matrix @ psi-values[0]*psi) < 1e-9
    return float(values[0]), psi


def reference(matrix, h):
    if h == 0:
        half = matrix.shape[0]//2
        direct = matrix[:half, :half]
        cross = matrix[:half, np.arange(matrix.shape[0]-1, half-1, -1)]
        even, vector = ground(direct+cross)
        odd, _ = ground(direct-cross)
        psi = np.concatenate([vector, vector[::-1]])/np.sqrt(2)
        assert np.linalg.norm(matrix @ psi-even*psi) < 1e-9
        assert odd >= even-1e-10
        return even, psi, dict(even_energy=even, odd_energy=odd, parity_gap=odd-even)
    energy, psi = ground(matrix)
    return energy, psi, {}


def measure(psi, matrix, exact, spins):
    psi = psi/np.linalg.norm(psi)
    p = psi**2
    applied = matrix @ psi
    energy = float(psi @ applied)
    c = correlations(p, spins)
    parity = float(psi @ psi[::-1])
    return dict(energy=energy, energy_error_per_spin=(energy-exact[0])/spins.shape[1],
                variance_per_spin=float(np.sum((applied-energy*psi)**2))/spins.shape[1],
                fidelity=float((psi @ exact[1])**2), parity_expectation=parity,
                odd_weight=(1-parity)/2, spin_flip_tv=float(np.abs(p-p[::-1]).sum()/2),
                site_magnetization=c['site_magnetization'],
                rms_site_magnetization=float(np.sqrt(np.mean(c['site_magnetization']**2))),
                q2=c['q2'], chi_connected=c['chi_connected'],
                probability_tv_to_exact=float(np.abs(p-exact[1]**2).sum()/2))


@torch.no_grad()
def amplitude(model, ham, gamma, spins):
    model.set_hamiltonian(ham, gamma)
    psi = model.log_amplitudes(torch.as_tensor(spins, dtype=torch.long)).exp().numpy()
    assert abs(np.sum(psi**2)-1) < 1e-9
    return psi/np.linalg.norm(psi)


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    directory = root/'data/kagome/transformer_af_shared_accuracy_v1'
    checkpoint = directory/'shared_alibi__L02-04-06-08__w064__d04__contexts24__seed401__final.pt'
    state = load_checkpoint(checkpoint, 'cpu')
    samples = {s:make_af_sample(2,s) for s in range(10)}
    # Check reconstructed disorder against the actual campaign inputs.
    for s in (0,1,2):
        with np.load(directory/f'disorder_L2_s{s}.npz') as saved:
            np.testing.assert_array_equal(samples[s].bonds,saved['bonds'])
            np.testing.assert_array_equal(-samples[s].J,saved['K'])
    model = UnifiedTransformer(samples[0].hamiltonian(.3),.5,
                               **state['metadata']['architecture']).double().eval()
    model.load_state_dict(state['model'])
    original = {k:v.clone() for k,v in model.state_dict().items()}
    spins = basis(12)
    contexts = [(s,g,h) for s in (0,1,2) for g,h in ((.5,0.),(.5,.3),(2.,.3))]
    contexts += [(s,.5,.3) for s in range(3,10)]
    rows, arrays, cached = [], {}, {}
    for s,g,h in contexts:
        ham = samples[s].hamiltonian(h)
        matrix = operator_matrix(ham,g)
        energy, exact, sectors = reference(matrix,h)
        assert np.min(exact) > 0
        psi = amplitude(model,ham,g,spins)
        states = dict(original=psi)
        if h == 0:
            states.update(even_projection=psi+psi[::-1],
                          probability_mixture=np.sqrt((psi**2+psi[::-1]**2)/2))
        measured = {name:measure(v,matrix,(energy,exact),spins) for name,v in states.items()}
        if h == 0:
            for name in ('even_projection','probability_mixture'):
                assert measured[name]['rms_site_magnetization'] < 1e-12
                assert abs(measured[name]['parity_expectation']-1) < 1e-12
            # Even two-spin observables are invariant under probability mixing.
            assert abs(measured['original']['q2']-measured['probability_mixture']['q2']) < 1e-12
        row = dict(sample_id=s,gamma=g,h=h,training_context=s in (0,1),
                   exact=measure(exact,matrix,(energy,exact),spins),sectors=sectors,states=measured)
        rows.append(row)
        key=f's{s}_g{g}_h{h}'
        arrays[key+'_amplitude']=psi
        arrays[key+'_exact']=exact
        if g == .5 and h == .3:
            cached[s]=(matrix,energy,exact,psi)
        print('ENUMERATED',s,g,h,'error/N',measured['original']['energy_error_per_spin'],
              'fidelity',measured['original']['fidelity'],flush=True)
    # Supply another sample's bonds while evaluating the physical target unchanged.
    # This controlled ablation tests conditioning; it is not a physical prediction.
    cross=[]
    for target in range(3):
        matrix,energy,exact,_=cached[target]
        for supplied in range(3):
            psi=cached[supplied][3]
            cross.append(dict(target=target,supplied=supplied,
                              **measure(psi,matrix,(energy,exact),spins)))
    # Translate bonds and spins together on the L2 torus; compare pulled-back p.
    translations=[]
    for s in range(3):
        ham=samples[s].hamiltonian(.3)
        lookup={tuple(b):i for i,b in enumerate(ham.bonds)}
        base=cached[s][3]**2
        for dx,dy in ((1,0),(0,1),(1,1)):
            perm=np.array([3*((((i//3)//2+dy)%2)*2+((i//3)%2+dx)%2)+i%3
                           for i in range(12)])
            new_J=np.empty_like(ham.J)
            for bond,value in zip(ham.bonds,ham.J):
                new_J[lookup[tuple(sorted(perm[bond]))]]=value
            from kagome_qea.hamiltonian import Hamiltonian
            translated=Hamiltonian(12,ham.bonds,new_J,ham.h)
            moved=np.empty_like(spins); moved[:,perm]=spins
            mapped=((1-moved)//2) @ (1 << np.arange(12))
            matrix=operator_matrix(translated,.5)
            np.testing.assert_allclose(matrix[mapped,:][:,mapped].toarray(),
                                       cached[s][0].toarray(),rtol=0,atol=1e-12)
            p=amplitude(model,translated,.5,spins)**2
            pulled=p[mapped]
            translations.append(dict(sample_id=s,dx=dx,dy=dy,
                                      probability_tv=float(np.abs(base-pulled).sum()/2),
                                      **measure(np.sqrt(pulled),cached[s][0],
                                                cached[s][1:3],spins)))
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    result=dict(checkpoint=checkpoint.name,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                precision='float64, CPU; all 4096 configurations',rows=rows,
                conditioning_ablation=cross,translations=translations,
                checks=['stored disorder matches reconstruction','wavefunction normalization',
                        'eigensolver residual < 1e-9','symmetrized local magnetizations vanish',
                        'probability mixture preserves q2','translated Hamiltonian equivalence',
                        'checkpoint weights unchanged'])
    write_json(directory/'symmetry_diagnostics.json',result)
    np.savez_compressed(directory/'symmetry_distributions.npz',**arrays)
    print('Saved diagnostics',flush=True)


if __name__ == '__main__':
    main()
