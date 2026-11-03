"""Reload a finished OBC checkpoint and audit numerical/projector interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .experiment import write_json
from .train_obc import load_obc, sample, FIELDS
from .physics import basis
from .vmc import local_energy
from .model import EnumeratedGuide
from .projector import project
from .guided_overlap_demo import histogram_tags


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(4)
    status = json.loads((root/'main/status.json').read_text())
    checkpoint = root/'main'/status['checkpoint']
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != status['sha256']:
        raise RuntimeError('Final checkpoint hash mismatch')
    model = load_obc(checkpoint, 'cuda').double()
    spins = torch.tensor(basis(12), device='cuda', dtype=torch.long)
    checks = []
    for gamma, h in FIELDS:
        ham = sample(2, 75000).hamiltonian(h)
        model.set_hamiltonian(ham, gamma)
        logs = model.log_amplitudes(spins).cpu().numpy()
        p = np.exp(2*logs)
        np.testing.assert_allclose(p.sum(), 1., atol=1e-10, rtol=0)
        if h == 0:
            np.testing.assert_allclose(logs, logs[::-1], atol=1e-10, rtol=0)
        selected = np.array([0, 73, 991, 4095])
        ratios = model.all_flip_log_ratios(spins[selected]).cpu().numpy()
        np.testing.assert_allclose(ratios, logs[selected[:,None] ^ (1 << np.arange(12))]-logs[selected,None], atol=1e-9, rtol=1e-9)
        psi = np.exp(logs)
        expected = operator_matrix(ham, gamma) @ psi / psi
        np.testing.assert_allclose(local_energy(model,ham,gamma,spins[selected]).cpu().numpy(), expected[selected], atol=1e-8, rtol=1e-9)
        draws = model.sample(12000, torch.Generator(device='cuda').manual_seed(1971)).cpu().numpy()
        for values, observed in [(basis(12)[:,0], draws[:,0]),
                                 (basis(12)[:,0]*basis(12)[:,1],draws[:,0]*draws[:,1])]:
            mean = p @ values
            se = np.sqrt(max(0.,p @ values**2-mean**2)/len(draws))
            if abs(observed.mean()-mean) > 6*se+.001:
                raise RuntimeError('Reloaded OBC sampling moment mismatch')
        result = project(ham,gamma,EnumeratedGuide(logs,12),n_walkers=32,n_replicas=4,
                         tau_block=.05,n_equilibrate=2,n_measure=2,tau_forward=.1,
                         n_cycles=2,seed=719,tagger=histogram_tags)
        np.testing.assert_allclose(np.asarray(result['pq_cycles']) @ np.linspace(-1,1,13)**2,
                                   result['q2_cycles'],atol=1e-12)
        checks.append(dict(gamma=gamma,h=h,normalization=float(p.sum()),passed=True))
    evaluation = json.loads((root/'main/evaluation.json').read_text())
    heldout = [r for r in evaluation['rows'] if r['L']==2 and 75000 <= r['sample_id'] < 75008]
    if len(heldout) != 24:
        raise RuntimeError('Missing final held-out contexts')
    summaries = []
    for gamma,h in FIELDS:
        rows = [r for r in heldout if r['gamma']==gamma and r['h']==h]
        summaries.append(dict(gamma=gamma,h=h,disorders=len(rows),
            mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rows])),
            max_energy_error_per_spin=max(r['energy_error_per_spin'] for r in rows),
            mean_fidelity=float(np.mean([r['fidelity'] for r in rows])),
            mean_probability_tv=float(np.mean([r['probability_tv_to_exact'] for r in rows]))))
    write_json(root/'final_verification.json',dict(passed=True,sha256=digest,checks=checks,
               heldout_summary=summaries,ground_state_qualified=False))
    lines = ['# OBC final checkpoint evaluation','',f'SHA256: `{digest}`','',
             'Reload, normalization, parity, flip ratios, local energy, sampling moments and GFMC histogram checks passed.',
             '', '| Gamma | h | Held-out disorders | Mean energy error/site | Mean fidelity | Mean probability TV |',
             '|---|---|---|---|---|---|']
    for r in summaries:
        lines.append(f"| {r['gamma']} | {r['h']} | {r['disorders']} | {r['mean_energy_error_per_spin']:.6g} | {r['mean_fidelity']:.6g} | {r['mean_probability_tv']:.6g} |")
    lines += ['', 'Exact comparisons cover L=2 and the three training field contexts on unseen disorder. Larger sizes have sampled variational diagnostics. GFMC ground-state convergence requires separate qualification.','']
    (root/'OBC_EVALUATION.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
