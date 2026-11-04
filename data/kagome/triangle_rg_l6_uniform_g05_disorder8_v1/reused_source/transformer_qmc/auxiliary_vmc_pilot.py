"""Finite L6 auxiliary-field pilot, with resumable VMC and independent screens."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .feedback_samplers import TwoCopyTempering
from .physics import make_af_sample, diagonal, basis
from .sampled_simple_guides import local_energy, energy_loss
from .simple_guides import RBM


SCHEDULES = dict(unpinned=[0.] * 6,
                 release=[.2, .1, .05, .02, 0., 0.], fixed=[.2] * 6)


def penalty(states, reference):
    return (states.shape[-1] - states @ reference) / 2


def atomic_save(path, state):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    temporary.replace(path)


def validate(root):
    """Exact pinned local energy/gradient and optimizer+sampler resumption."""
    from scipy.sparse import diags
    from kagome_qea.exact import operator_matrix
    ham = make_af_sample(2, 60000).hamiltonian(.15)
    x = torch.tensor(basis(12), dtype=torch.float64)
    model = RBM(12, 2, 7)
    ref = x[173]; matrix = operator_matrix(ham, .6)
    for epsilon in [0., .2]:
        d = penalty(x, ref)
        np.testing.assert_array_equal(d.numpy(), (x != ref).sum(1).numpy())
        full = matrix + diags(epsilon * d.numpy())
        logs = model(x); amplitude = torch.exp(logs-logs.max())
        p = amplitude.square(); p = p/p.sum()
        energies = local_energy(model, x, ham, .6) + epsilon*d
        exact = full @ amplitude.detach().numpy() / amplitude.detach().numpy()
        np.testing.assert_allclose(energies.numpy(), exact, atol=1e-10)
        # Differentiate the symmetric quadratic directly through a sparse multiply.
        full_t = torch.tensor(full.toarray(), dtype=torch.float64)
        objective = amplitude @ (full_t @ amplitude) / amplitude.square().sum() / 12
        direct = torch.autograd.grad(objective, tuple(model.parameters()))
        logs = model(x)
        covariance = 2*(p.detach()*(logs*(energies-(p.detach()*energies).sum()))).sum()/12
        actual = torch.autograd.grad(covariance, tuple(model.parameters()))
        for a, b in zip(direct, actual):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    # Bitwise continuation on CUDA, including Adam and replica labels/RNG.
    def create():
        m = RBM(12, 2, 11).cuda()
        g = torch.Generator(device='cuda').manual_seed(271)
        s = TwoCopyTempering(m, 16, 4, g)
        o = torch.optim.Adam(m.parameters(), lr=.001)
        return m, g, s, o
    def step(m, g, s, o):
        states = s.sweep(); energies = local_energy(m, states, ham, .6)
        o.zero_grad(); energy_loss(m(states), energies, 12).backward(); o.step()
    m, g, s, o = create(); step(m, g, s, o)
    path = root/'resume_validation.pt'
    atomic_save(path, dict(model=m.state_dict(), optimizer=o.state_dict(), sampler=s.state_dict(), rng=g.get_state()))
    step(m, g, s, o)
    m2, g2, s2, o2 = create(); saved = torch.load(path, weights_only=False)
    m2.load_state_dict(saved['model']); o2.load_state_dict(saved['optimizer'])
    s2 = TwoCopyTempering(m2, 16, 4, g2, saved['sampler']); g2.set_state(saved['rng'].cpu())
    step(m2, g2, s2, o2)
    for a, b in zip(m.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key, value in s.state_dict().items():
        other = s2.state_dict()[key]
        if isinstance(value, torch.Tensor): torch.testing.assert_close(value, other, atol=0, rtol=0)
        else: assert value == other
    torch.testing.assert_close(g.get_state(), g2.get_state(), atol=0, rtol=0)
    write(root/'validation.json', dict(passed=True, exact_local_energy=True,
        exact_covariance_gradient=True, resumed_cuda_continuation_bitwise=True,
        source_sha256=sha(Path(__file__))))


def candidate_bank(root, sample):
    path = root/'candidate_bank.npz'; info = root/'candidate_bank.json'
    if path.exists():
        record = json.loads(info.read_text()); assert sha(path) == record['raw_sha256']
        with np.load(path) as raw: refs = raw['references'].copy()
        if refs.shape != (3, sample.hamiltonian(.15).n_sites):
            raise RuntimeError('Saved candidate bank has insufficient references; no training launched')
        return refs
    ham = sample.hamiltonian(.15); n = ham.n_sites; restarts = 128
    entropy = [20261104, 6, sample.sample_id, 0]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    states = (2*rng.integers(2, size=(restarts,n))-1).astype(np.int8)
    initial = states.copy(); coupling = np.zeros((n,n))
    a,b = ham.bonds.T; coupling[a,b] = -ham.J; coupling[b,a] = -ham.J
    fields = states@coupling; best = states.copy(); best_e = diagonal(ham, states)
    temperatures = np.geomspace(2., .02, 64); history=[]
    for temperature in temperatures:
        for _ in range(16):
            for i in rng.permutation(n):
                delta = 2*states[:,i]*(ham.h-fields[:,i])
                take = np.log(rng.random(restarts)) < -delta/temperature
                change = -2*states[:,i]*take
                fields += change[:,None]*coupling[i]
                states[:,i] += change
            energy = diagonal(ham, states); improve = energy < best_e
            best[improve] = states[improve]; best_e[improve] = energy[improve]
            history.append(energy.copy())
    prequench = np.concatenate([best, states]); quenched = prequench.copy()
    fields = quenched@coupling
    for sweep in range(108):
        accepted = 0
        for i in rng.permutation(n):
            delta = 2*quenched[:,i]*(ham.h-fields[:,i]); take = delta < -1e-12
            accepted += take.sum(); change = -2*quenched[:,i]*take
            fields += change[:,None]*coupling[i]; quenched[:,i] += change
        if accepted == 0: break
    residual = (2*quenched*(ham.h-quenched@coupling) < -1e-10).sum(1)
    bank = np.unique(np.concatenate([prequench,quenched]),axis=0)
    energies = diagonal(ham, bank)
    # np.unique sorts lexicographically; stable sort resolves exact ties.
    order = np.argsort(energies,kind='stable'); selected=[]
    for i in order:
        if energies[i] > energies[order[0]]+2.16: break
        if all(np.count_nonzero(bank[i]!=bank[j]) >= 11 for j in selected): selected.append(int(i))
        if len(selected)==3: break
    refs=bank[selected]
    np.savez_compressed(path, points=sample.points,bonds=sample.bonds,J=sample.J,
        initial=initial,prequench=prequench,quenched=quenched,bank=bank,energies=energies,
        references=refs,selected=selected,history=history,temperatures=temperatures,residual_downhill=residual)
    write(info,dict(sample_id=sample.sample_id,seed_entropy=entropy,restarts=restarts,
        rng_policy='Independent restart coordinates in one fixed vectorized random stream; shared sweep permutations.',
        reference_energies=energies[selected].tolist(),minimum_bank_energy=float(energies.min()),
        unique_candidates=len(bank),reference_distances=(refs[:,None]!=refs[None,:]).sum(2).tolist(),
        reference_count=len(refs),quench_sweeps=sweep+1,residual_downhill=int(residual.sum()),raw_sha256=sha(path)))
    if len(refs)!=3: raise RuntimeError('Insufficient diverse references; bank retained, no training launched')
    return refs


def train_block(args, root, branch, reference, target):
    folder=root/'training'/branch; folder.mkdir(parents=True,exist_ok=True)
    snapshot=folder/f'update_{target:04d}.pt'
    signature=dict(manifest_sha256=sha(root/'manifest.json'),branch=branch,reference=reference.tolist())
    if snapshot.exists():
        saved=torch.load(snapshot,map_location='cpu',weights_only=False)
        assert saved['signature']==signature and saved['step']==target
        return snapshot
    cp=folder/'resume.pt'; model=RBM(108,2,args.seed).cuda()
    generator=torch.Generator(device='cuda').manual_seed(20261104+args.seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]; step=0
    if cp.exists():
        saved=torch.load(cp,map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        sampler=TwoCopyTempering(model,128,16,generator,saved['sampler'])
        generator.set_state(saved['rng'].cpu()); step=saved['step'];history=saved['history']
        assert saved['signature']==signature and step<=target
    else:
        saved=torch.load(args.initial,map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);sampler=TwoCopyTempering(model,128,16,generator)
        for _ in range(400):sampler.sweep()
    ham=make_af_sample(6,args.sample_id).hamiltonian(.15)
    ref=torch.tensor(reference,dtype=torch.float64,device='cuda')
    arm=branch.split('_r')[0]; started=time.monotonic()
    while step<target:
        epsilon=SCHEDULES[arm][step//400]
        for _ in range(10):states=sampler.sweep()
        e0=local_energy(model,states,ham,.6); e=e0+epsilon*penalty(states,ref)
        optimizer.zero_grad();energy_loss(model(states),e,108).backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();step+=1
        if step%50==0:
            row=dict(step=step,epsilon=epsilon,target_energy_per_site=float(e0.mean()/108),
                target_variance_per_site=float(e0.var(unbiased=False)/108),
                training_energy_per_site=float(e.mean()/108),gradient_norm=float(norm),
                seconds_this_block=time.monotonic()-started)
            history.append(row)
            atomic_save(cp,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                sampler=sampler.state_dict(),rng=generator.get_state(),step=step,history=history,
                signature=signature))
            write(folder/'history.json',dict(branch=branch,rows=history,transport=sampler.diagnostics()))
            write(root/'status.json',dict(state='training',branch=branch,**row,utc=datetime.now(timezone.utc).isoformat()))
            print(branch, row, flush=True)
    if not snapshot.exists():atomic_save(snapshot,dict(model=model.state_dict(),step=step,
        signature=signature,epsilon=SCHEDULES[arm][(step-1)//400]))
    return snapshot


@torch.no_grad()
def frozen(args, root, branch, checkpoint, references, update):
    folder=root/'frozen'/branch;folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'update_{update:04d}.json'
    if dest.exists():
        r=json.loads(dest.read_text());assert r['model_sha256']==sha(checkpoint)
        assert r['raw_sha256']==sha(dest.with_suffix('.npz'))
        return r
    model=RBM(108,2).cuda();model.load_state_dict(torch.load(checkpoint,weights_only=False)['model'])
    ham=make_af_sample(6,args.sample_id).hamiltonian(.15);rows=[];arrays={}
    refs=torch.tensor(references,dtype=torch.float64,device='cuda')
    for index,start in enumerate(['random','reference','inverse']):
        g=torch.Generator(device='cuda').manual_seed(300000+100*update+index)
        sampler=TwoCopyTempering(model,128,16,g)
        # Cycle through all three bank references across independent columns.
        if index:sampler.states[:]=((-1 if index==2 else 1)*refs[torch.arange(128,device='cuda')%3])[None]
        energies=[];spins=[]
        for sweep in range(400+1024):
            x=sampler.sweep()
            if sweep>=400 and (sweep-400)%4==0:
                energies.append(local_energy(model,x,ham,.6).cpu().numpy())
                spins.append(x.cpu().numpy().astype(np.int8))
        energy=np.array(energies);states=np.array(spins);flat=states.reshape(-1,108).astype(float)
        magnetization=flat.mean(0);pair=flat.T@flat/len(flat);connected=pair-np.outer(magnetization,magnetization)
        overlap=flat@references.T/108
        row=dict(start=start,energy_per_site=float(energy.mean()/108),variance_per_site=float(energy.var()/108),
            chain_mean_energy_sd_per_site=float(energy.mean(0).std()/108),
            energy_chain_standard_error_per_site=float(energy.mean(0).std(ddof=1)/108/np.sqrt(128)),
            reference_overlap=overlap.mean(0).tolist(),q2_born=float((pair**2).mean()),
            chi_connected_born=float((connected**2).sum()/108),unique_states=len(np.unique(states.reshape(-1,108),axis=0)),
            **sampler.diagnostics())
        rows.append(row);arrays[start+'_energy']=energy;arrays[start+'_spins']=states
        arrays[start+'_connected']=connected;arrays[start+'_reference_overlap']=overlap
    np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    result=dict(branch=branch,update=update,rows=rows,model_sha256=sha(checkpoint),raw_sha256=sha(dest.with_suffix('.npz')),
        energy_start_range=float(np.ptp([r['energy_per_site'] for r in rows])),
        scope='Born-distribution frozen-guide diagnostics, not projected ground-state estimates')
    write(dest,result);return result


def report(root):
    frozen_rows=[json.loads(p.read_text()) for p in sorted((root/'frozen').glob('*/*.json'))]
    projections=[json.loads(p.read_text()) for p in sorted((root/'projection').glob('*/runs/*.json'))]
    data=dict(updated_utc=datetime.now(timezone.utc).isoformat(),frozen=frozen_rows,projections=projections,
              ground_state_qualified=False)
    write(root/'early_results.json',data)
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    cells=[nbformat.v4.new_markdown_cell(r'''# L6 auxiliary-field VMC: early measurements

One disorder (60000), one training seed (0), three classical references.
All seven arms share the same high-field VMC initialization; each receives
2400 target-field VMC updates. Release uses 400 updates each at
$\epsilon=0.2,0.1,0.05,0.02$, then 800 at zero. Unpinned and fixed controls
receive the same budget. The blocked two-copy tempering kernel has cold
distribution $|\psi|^2$ and a different hot path from power tempering.

Frozen diagnostics sample each guide's Born distribution. Their energy,
variance, correlations and start dependence assess guide quality and sampling;
they are not ground-state measurements. All projection results use the
original Hamiltonian, $\Gamma=0.6$, $h=0.15$, with no auxiliary field.
Projection settings: four 512-walker populations, equilibration 16,
forward time 16, eight cycles, two repeats per guide. Retain all genealogy
alarms. Two repeats and one disorder cannot establish convergence or generality.
'''),nbformat.v4.new_code_cell('import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n'
        f'data=json.loads({json.dumps(json.dumps(data))})\n'
        'print("Updated UTC:",data["updated_utc"])\n'
        'rows=[dict(branch=g["branch"],update=g["update"],**r) for g in data["frozen"] for r in g["rows"]]\n'
        'df=pd.DataFrame(rows)\n'
        'if len(df):\n'
        '    display(df[["branch","update","start","energy_per_site","variance_per_site","q2_born","chi_connected_born","reference_overlap","round_trips"]])\n'
        '    means=df.groupby(["branch","update"])[["energy_per_site","variance_per_site"]].mean().reset_index()\n'
        '    fig,axes=plt.subplots(1,2,figsize=(12,4))\n'
        '    for branch,r in means.groupby("branch"):\n'
        '        axes[0].plot(r["update"],r.energy_per_site,"o-",label=branch)\n'
        '        axes[1].plot(r["update"],r.variance_per_site,"o-",label=branch)\n'
        '    axes[0].set(ylabel="Target energy / site",xlabel="VMC updates")\n'
        '    axes[1].set(ylabel="Target local-energy variance / N",xlabel="VMC updates")\n'
        '    axes[0].legend(fontsize=7);fig.tight_layout();plt.show()\n'
        'pr=[dict(branch=r["task"]["branch"],repeat=r["task"]["repeat"],energy_per_site=r["energy"]/108,q2=r["q2"],ess=r["minimum_descendant_ess"],largest_lineage=r["largest_lineage"],alarm=r["ancestry_alarm"]) for r in data["projections"]]\n'
        'display(pd.DataFrame(pr))\n'
        'if not pr:print("Independent projection results pending.")\n')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')))
    NotebookClient(nb,timeout=120,resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(nb,root/'auxiliary_field_vmc_results.ipynb')
    html,_=HTMLExporter().from_notebook_node(nb);(root/'auxiliary_field_vmc_results.html').write_text(html)
    return data


def run(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    sample=make_af_sample(6,args.sample_id)
    initial=torch.load(args.initial,map_location='cpu',weights_only=False)
    assert initial['signature']['task']==dict(L=6,gamma=4.,sample_id=args.sample_id,kind='rbm_2',seed=args.seed,mode='direct')
    sources={str(p.relative_to(Path(__file__).parents[1])):sha(p) for p in Path(__file__).parents[1].rglob('*.py')}
    manifest=dict(sample_id=args.sample_id,seed=args.seed,initial=str(args.initial),initial_sha256=sha(args.initial),
        schedules=SCHEDULES,updates_per_stage=400,chains=128,temperatures=16,sweeps_per_update=10,
        sampler='TwoCopyTempering: exact cold Born marginal, blocked Gibbs; differs from draft power ladder',
        optimizer='Fresh Adam .001, cap1; moments/sampler persist across release',
        sources=sources,workers=1,deadline=None,planned_branches=7,planned_projections=14,
        scope='First-disorder first-seed L6 diagnostic, no ground-state qualification')
    path=root/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write(path,manifest)
    if args.validate_only:
        validate(root)
        return
    validation=json.loads((root/'validation.json').read_text())
    assert validation['passed'] and validation['source_sha256']==sha(Path(__file__))
    references=candidate_bank(root,sample)
    branches=[('unpinned',0)]+[(f'{arm}_r{r}',r) for arm in ['release','fixed'] for r in range(3)]
    failures=json.loads((root/'failures.json').read_text()) if (root/'failures.json').exists() else []
    for target in [400,800,1200,1600,2000,2400]:
        for branch,ref in branches:
            if any(f['branch']==branch for f in failures):continue
            try:
                cp=train_block(args,root,branch,references[ref],target)
                if target in [400,1600,2000,2400]:frozen(args,root,branch,cp,references,target)
            except Exception:
                failures.append(dict(branch=branch,target=target,traceback=traceback.format_exc()))
                write(root/'failures.json',failures);print(failures[-1]['traceback'],flush=True)
        if target==400 or target>=1600:report(root)
    from .projected_overlap_histograms import run_one
    from .feedback_frontier import verify_measurement
    for repeat in [0,1]:
        for branch,ref in branches:
            if any(f['branch']==branch for f in failures):continue
            folder=root/'projection'/branch
            task=dict(L=6,gamma=.6,sample_id=args.sample_id,kind='rbm_2',training_seed=args.seed,
                audit='auxiliary_vmc_pilot_v1',branch=branch,walkers=512,equil=16.,forward=16.,
                start='guide',delta=.1,cycles=8,repeat=repeat)
            write(root/'status.json',dict(state='projecting',branch=branch,repeat=repeat,utc=datetime.now(timezone.utc).isoformat()))
            try:
                row=run_one(dict(task=task,model=str(root/'training'/branch/'update_2400.pt'),projection_device='cuda'),folder)
                verification=verify_measurement(row,folder)
                write(folder/f'verification_{repeat}.json',verification)
            except Exception:
                failures.append(dict(branch=branch,repeat=repeat,traceback=traceback.format_exc()))
                write(root/'failures.json',failures);print(failures[-1]['traceback'],flush=True)
            report(root)
    write(root/'status.json',dict(state='completed_unqualified',failures=failures,utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--sample-id',type=int,default=60000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a)
