"""Finite feedback ablations: exact L2 continuation screens before matched L6.

One worker, no deadline, no retries or selected-away contexts. No larger sizes.
Source snapshots are immutable after launch; resumption validates manifests.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .disorder_scaling import observables as raw_observables
from .parallel_disorder_scaling import fit_name
from .physics import basis, exact_reference, make_af_sample
from .production_protocol import geometry, identity
from .projected_overlap_histograms import run_one, histogram
from .sampled_simple_guides import freeze_diagnostics, sampling_alarms
from .self_learning_l6_pilot import contexts, measurement_task, copy_verified
from .self_learning_measurements import observables, compare, frozen_sampler
from .self_learning_rbm import run as learn, exact_evaluation
from .feedback_samplers import TwoCopyTempering
from .simple_guides import RBM


def variants():
    base = dict(walkers=1024, islands=1, negative_temperatures=1, step_rms=0., step_max=0.)
    pt = dict(base, negative_temperatures=8)
    bounded = dict(pt, step_rms=.025, step_max=.15)
    return dict(gibbs_long=base, tempered=pt, tempered_bounded=bounded,
                tempered_bounded_large=dict(bounded, walkers=4096),
                tempered_bounded_islands=dict(bounded, islands=4))


def settings(variant, L, sid, seed, initial, output):
    return argparse.Namespace(output=output, initial=initial, L=L, gamma=.6,
        sample_id=sid, seed=seed, stints=6 if L == 2 else 12,
        equil=8., collect=12., delta=.1, sample_walkers=64, updates=1000,
        batch=256, gibbs=10, initial_sweeps=400, lr=.001, device='cuda', **variants()[variant])


def l2_task(seed, repeat):
    # Match the original final-L2 measurement task identity, hence random streams.
    return dict(L=2, gamma=.6, sample_id=60000, kind='rbm_2', training_seed=seed,
        audit='self_learning_final_l2', walkers=512, equil=16., forward=16.,
        start='guide', delta=.1, cycles=8, repeat=repeat)


def verify_measurement(record, folder):
    """Reconstruct physical outputs and genealogy from raw tagged populations."""
    task = record['task']; path = folder/'runs'/(identity(task)[:24]+'.npz')
    assert sha(path) == record['raw_sha256']
    with np.load(path) as raw:
        tags = raw['tags'].astype(float); counts = raw['counts']; w = task['walkers']
        assert tags.shape == (task['cycles'],4,w,3*task['L']**2)
        assert np.isin(tags,[-1,1]).all() and (counts >= 0).all()
        np.testing.assert_array_equal(counts.sum(2), np.full(counts.shape[:2], w))
        grid, mass = histogram(tags,counts)
        np.testing.assert_allclose(mass,record['mass'],atol=1e-12,rtol=0)
        means = np.einsum('crw,crwn->crn',counts/w,tags)
        pairs = np.einsum('crw,crwi,crwj->crij',counts/w,tags,tags)
        values = raw_observables(means.mean(0),pairs.mean(0),task['L'])
        for key in ['q','q2','spectrum']:
            np.testing.assert_allclose(values[key],record[key],atol=1e-12,rtol=0)
        ess = w*w/(counts.astype(float)**2).sum(2); largest = counts.max(2)/w
        np.testing.assert_allclose([ess.min(),largest.max()],
            [record['minimum_descendant_ess'],record['largest_lineage']],atol=1e-12)
        energy = raw['energy'][round(task['equil']/task['delta']):].mean()
        np.testing.assert_allclose(energy,record['energy'],atol=1e-12)
        assert record['ancestry_alarm'] == bool(ess.min()<8 or largest.max()>.5)
    return dict(raw_sha256=sha(path),reconstructed=True)


@torch.no_grad()
def audit_training(folder, L, sid):
    """Evaluate data fit, negative-chain discrepancy and population disagreement.

Shared-target averages do not establish generalization or chain equilibration.
Independent islands retain equal total mass; lineage alarms are never reweighted.
"""
    manifest = json.loads((folder/'manifest.json').read_text()); cfg = manifest['settings']
    rows = json.loads((folder/'summary.json').read_text())['rows']; audited = []
    previous = None
    ham = make_af_sample(L,sid).hamiltonian(.15)
    for row in rows:
        path = folder/f"stint_{row['stint']:03d}"
        for name,key in [('input_model.pt','input_model_sha256'),('model.pt','model_sha256'),
                         ('walkers.npz','raw_sha256'),('fit.npz','fit_sha256')]:
            assert sha(path/name) == row[key]
        raw = np.load(path/'walkers.npz'); fit = np.load(path/'fit.npz')
        data = raw['samples']; neg = fit['negative_samples']
        assert data.shape == (round(cfg['collect']/cfg['delta'])*cfg['sample_walkers'],3*L*L)
        for x in [data,neg]:
            assert np.isin(x,[-1,1]).all()
        for x in [raw,fit]:
            for name in x.files:
                assert np.isfinite(x[name]).all(),name
        assert fit['gradient_trace'][-1,0] == cfg['updates']
        lineage = raw['lineages'][round(cfg['equil']/cfg['delta']):].reshape(-1,2)
        np.testing.assert_allclose([lineage[:,0].min(),lineage[:,1].max()],
            [row['minimum_training_lineage_ess'],row['largest_training_lineage']],atol=1e-12)
        models = []
        for name in ['input_model.pt','model.pt']:
            model = RBM(3*L*L,2)
            model.load_state_dict(torch.load(path/name,map_location='cpu',weights_only=False)['model'])
            models.append(model)
        if previous is not None:
            for key,value in models[0].state_dict().items():
                torch.testing.assert_close(value,previous[key],atol=0,rtol=0)
        previous = models[1].state_dict()
        x = torch.tensor(data,dtype=torch.float64); n = torch.tensor(neg,dtype=torch.float64)
        change = 2*(models[1](x)-models[0](x));change -= change.mean()
        item = dict(stint=row['stint'],unique_configurations=len(np.unique(data,axis=0)),
            empirical_log_density_change_rms=float(change.square().mean().sqrt()),
            positive_negative_spin_rms=float(np.sqrt(np.mean((data.mean(0)-neg.mean(0))**2))),
            minimum_lineage_ess=float(lineage[:,0].min()),largest_lineage=float(lineage[:,1].max()))
        if 'sample_island' in raw and cfg.get('islands',1)>1:
            means = np.asarray([data[raw['sample_island']==k].mean(0) for k in range(cfg['islands'])])
            item['between_island_spin_rms'] = float(np.std(means,axis=0).mean())
        if L == 2:
            for label,model in zip(['before','after'],models):
                evaluation = exact_evaluation(model,ham,.6,data)
                for key,value in evaluation.items():
                    np.testing.assert_allclose(value,row[label][key],atol=1e-10,rtol=1e-10)
                item[label] = evaluation
            all_states=torch.tensor(basis(12),dtype=torch.float64)
            exact=exact_reference(ham,.6)
            before_logs=models[0](all_states).numpy()
            ideal=np.exp(before_logs-before_logs.max())*exact['amplitude'];ideal/=ideal.sum()
            ids=((1-data.astype(np.int64))//2)@(1 << np.arange(12))
            empirical=np.bincount(ids,minlength=4096)/len(data)
            item['empirical_vs_ideal_mixed_tv']=float(abs(empirical-ideal).sum()/2)
            p=torch.softmax(2*models[1](all_states),0)
            model=models[1];h=torch.tanh(all_states@model.w+model.b);hn=torch.tanh(n@model.w+model.b)
            exact_gradient=torch.cat((2*(p@all_states),2*(p@h),(2*all_states.T@(p[:,None]*h)).flatten()))
            negative_gradient=torch.cat((2*n.mean(0),2*hn.mean(0),(2*n.T@hn/len(n)).flatten()))
            item['negative_gradient_exact_rms']=float((negative_gradient-exact_gradient).square().mean().sqrt())
        audited.append(item)
    write(folder/'audit.json',dict(rows=audited,ground_state_qualified=False))
    return audited


@torch.no_grad()
def frozen_tempered_l2(model, probability, seed, output):
    dest = output/'frozen_tempered.json'
    if dest.exists():
        result=json.loads(dest.read_text());assert result['raw_sha256']==sha(dest.with_suffix('.npz'))
        return result
    device=next(model.parameters()).device
    x=torch.tensor(basis(12),dtype=torch.float64,device=device)
    p=torch.tensor(probability,dtype=torch.float64,device=device)
    h=torch.tanh(x@model.w+model.b)
    truth=torch.cat((2*(p@x),2*(p@h),(2*x.T@(p[:,None]*h)).flatten())).cpu().numpy()
    rows=[];arrays={'exact_gradient':truth}
    for k,start in enumerate(['random','up','down']):
        generator=torch.Generator(device=device).manual_seed(20261105+seed*100+k)
        sampler=TwoCopyTempering(model,256,8,generator)
        if start!='random':sampler.states.fill_(1 if start=='up' else -1)
        statistics=torch.zeros((256,len(truth)),dtype=torch.float64,device=device);samples=[]
        for step in range(1500):
            s=sampler.sweep()
            if step>=500 and (step-500)%10==0:
                samples.append(s.cpu().numpy().astype(np.int8));h=torch.tanh(s@model.w+model.b)
                statistics+=torch.cat((2*s,2*h,(2*s[:,:,None]*h[:,None,:]).flatten(1)),dim=1)
        chain_means=(statistics/len(samples)).cpu().numpy()
        estimate=chain_means.mean(0);se=chain_means.std(0,ddof=1)/np.sqrt(256)
        arrays[start]=np.asarray(samples);arrays[start+'_gradient_mean']=estimate;arrays[start+'_gradient_se']=se
        rows.append(dict(start=start,gradient_rms_error=float(np.sqrt(np.mean((estimate-truth)**2))),
            gradient_max_standardized_error=float(np.max(np.abs(estimate-truth)/np.maximum(se,1e-15))),
            **sampler.diagnostics()))
    np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    result=dict(rows=rows,raw_sha256=sha(dest.with_suffix('.npz')),ground_state_qualified=False)
    write(dest,result);return result


def screen(rows):
    problems=[]
    if len(rows)!=2 or {r['seed'] for r in rows}!={0,1}:problems.append('missing or duplicate prescribed seed')
    for row in rows:
        if not row['audit_complete']:problems.append(f"seed{row['seed']}: incomplete training audit")
        for measurement in row['measurements']:
            e=measurement['errors']
            finite=np.isfinite([e[k] for k in ['energy_error_per_site','q2_error','histogram_tv','spectrum_max_tolerance_ratio']]).all()
            if (not finite or measurement['ancestry_alarm'] or abs(e['energy_error_per_site'])>.001
                or abs(e['q2_error'])>.01 or e['histogram_tv']>.05 or e['spectrum_max_tolerance_ratio']>1):
                problems.append(f"seed{row['seed']}/repeat{measurement['task']['repeat']}: physical screen failed")
        if len(row['measurements'])!=2 or {m['task']['repeat'] for m in row['measurements']}!={0,1}:problems.append('missing or duplicate measurement repeat')
        if len(row['tempered']['rows'])!=3 or {r['start'] for r in row['tempered']['rows']}!={'random','up','down'}:
            problems.append('missing or duplicate frozen start')
        for result in row['tempered']['rows']:
            finite=np.isfinite([result['gradient_rms_error'],*result['exchange_fraction']]).all()
            if not finite or result['round_trips']==0 or min(result['exchange_fraction'])<.01 or result['gradient_rms_error']>.02:
                problems.append(f"seed{row['seed']}/{result['start']}: frozen tempered screen failed")
    return dict(pass_diagnostic_screen=not problems,problems=problems,
        uncertainty='Finite L2 implementation/observable screen only. No population, projection-time or chain-bias bound.',
        ground_state_qualified=False)


def run(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    regression=json.loads(args.regressions.read_text())
    assert regression['passed'] and regression['tests_run']>=12 and regression['skipped']==0
    for relative,digest in regression['source_hashes'].items():
        assert sha(Path(__file__).parent/relative)==digest,relative
    sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    inputs={}
    for seed in [0,1]:
        source=args.l2/f'seed{seed}/stint_005/model.pt'
        destination=root/'controls'/f'l2_seed{seed}.pt';copy_verified(source,destination)
        inputs[f'l2_seed{seed}']=dict(path=str(destination),sha256=sha(source))
    for task in contexts():
        name=fit_name(task);source=args.l6/'controls'/name/(name+'.pt')
        destination=root/'controls'/name/(name+'.pt');copy_verified(source,destination)
        for path in sorted((source.parent/'runs').glob('*')):
            copy_verified(path,destination.parent/'runs'/path.name)
        controls=list((destination.parent/'runs').glob('*.json'));assert len(controls)==2
        for path in controls:
            record=json.loads(path.read_text())
            assert record['provenance']['model_sha256']==sha(source)
            assert sha(path.with_suffix('.npz'))==record['raw_sha256']
            assert record['task'] in [measurement_task(task,r) for r in [0,1]]
        inputs[name]=dict(path=str(destination),sha256=sha(source))
    manifest=dict(format=1,variants=variants(),inputs=inputs,sources=sources,
        source_hashes=regression['source_hashes'],regression_sha256=sha(args.regressions),
        l2_settings={k:v for k,v in vars(settings('gibbs_long',2,60000,0,None,None)).items()
                     if k not in ['output','initial']},
        L2='All five arms, both seeds. Six continuation stints from archived original sixth-stint guides.',
        L6='Only after arm-specific L2 screen passes for both seeds: both disorders and seeds, 12 stints from same VMC-400 controls.',
        l2_stints=60,l2_measurements=20,max_l6_stints=240,max_l6_measurements=40,
        measurement='Matched final-guide four-population settings and RNG task identities; no adaptive tags.',
        workers=1,deadline=None,retries=0,ground_state_qualified=False,
        resource_caveat='Same likelihood updates/samples, unequal compute. Large and islands arms both use 4096 walkers total; report timings.',
        scope='Finite Gamma=.60 L2/L6 diagnostic only; no broad disorder expansion, no L8/L10, no automatic qualification.')
    path=root/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write(path,manifest)
    failures=[];l2rows=[];l6rows=[];gates={}
    def status(state,current=None):
        write(root/'status.json',dict(state=state,current=current,updated_utc=datetime.now(timezone.utc).isoformat(),
            completed_l2_contexts=len(l2rows),planned_l2_contexts=10,completed_l6_contexts=len(l6rows),
            maximum_l6_contexts=20,failures=failures,gates=gates,deadline=None,ground_state_qualified=False))
    def failure(variant,L,sid,seed,error):
        failures.append(dict(variant=variant,L=L,sample_id=sid,seed=seed,error=repr(error),traceback=traceback.format_exc()))
        write(root/'failures.json',failures)
    ham=make_af_sample(2,60000).hamiltonian(.15);states=basis(12);truth=exact_reference(ham,.6)
    truth=dict(energy=truth['energy'],residual=truth['residual'],**observables(truth['probability'],states,geometry(2)))
    write(root/'exact_l2.json',truth)
    # Complete ALL small-system arms before dispatching any large-system extension.
    for variant in variants():
        for seed in [0,1]:
            folder=root/'l2'/variant/f'seed{seed}'
            try:
                status('l2_adapting',dict(variant=variant,seed=seed))
                learn(settings(variant,2,60000,seed,Path(inputs[f'l2_seed{seed}']['path']),folder/'training'))
                audit=audit_training(folder/'training',2,60000)
                final=folder/'training/stint_005/model.pt';model=RBM(12,2).to('cuda')
                model.load_state_dict(torch.load(final,map_location='cuda',weights_only=False)['model'])
                with torch.no_grad():
                    probability=torch.softmax(2*model(torch.tensor(states,dtype=torch.float64,device='cuda')),0).cpu().numpy()
                status('l2_frozen_checks',dict(variant=variant,seed=seed))
                gibbs=frozen_sampler(model,probability,states,seed,folder)
                tempered=frozen_tempered_l2(model,probability,seed,folder)
                measured=[]
                for repeat in [0,1]:
                    status('l2_projecting',dict(variant=variant,seed=seed,repeat=repeat))
                    row=run_one(dict(model=str(final),task=l2_task(seed,repeat)),folder/'projector')
                    row['verification']=verify_measurement(row,folder/'projector')
                    row['errors']=dict(energy_error_per_site=(row['energy']-truth['energy'])/12,**compare(row,truth))
                    measured.append(row)
                result=dict(variant=variant,seed=seed,model_sha256=sha(final),audit_complete=len(audit)==6,
                    exact_guide=exact_evaluation(model,ham,.6),gibbs=gibbs,tempered=tempered,measurements=measured)
                write(folder/'result.json',result);l2rows.append(result)
                print(json.dumps(dict(phase='l2_done',variant=variant,seed=seed,exact_guide=result['exact_guide'])),flush=True)
            except Exception as error:failure(variant,2,60000,seed,error)
            write(root/'l2_results.json',dict(rows=l2rows,ground_state_qualified=False))
        gates[variant]=screen([r for r in l2rows if r['variant']==variant])
        write(root/'l2_gates.json',gates)
    for variant in variants():
        if not gates[variant]['pass_diagnostic_screen']:continue
        for task in contexts():
            name=fit_name(task);folder=root/'l6'/variant/name
            try:
                status('l6_adapting',dict(variant=variant,**task))
                learn(settings(variant,6,task['sample_id'],task['seed'],Path(inputs[name]['path']),folder/'training'))
                audit_training(folder/'training',6,task['sample_id'])
                final=folder/'training/stint_011/model.pt';frozen=folder/'frozen.json'
                status('l6_frozen_checks',dict(variant=variant,**task))
                if frozen.exists():
                    diagnostic=json.loads(frozen.read_text())
                    assert diagnostic['model_sha256']==sha(final) and diagnostic['raw_sha256']==sha(frozen.with_suffix('.npz'))
                else:
                    model=RBM(108,2).to('cuda')
                    model.load_state_dict(torch.load(final,map_location='cuda',weights_only=False)['model'])
                    checks,raw=freeze_diagnostics(model,make_af_sample(6,task['sample_id']).hamiltonian(.15),
                                                  .6,256,8,200,20261104+task['seed'])
                    np.savez_compressed(frozen.with_suffix('.npz'),**raw)
                    diagnostic=dict(rows=checks,sampling_alarms=sampling_alarms(checks,256),
                        model_sha256=sha(final),raw_sha256=sha(frozen.with_suffix('.npz')),ground_state_qualified=False)
                    write(frozen,diagnostic)
                measured=[]
                # Sampling alarms do not exclude physical diagnostic measurements.
                for repeat in [0,1]:
                    status('l6_projecting',dict(variant=variant,repeat=repeat,**task))
                    row=run_one(dict(model=str(final),task=measurement_task(task,repeat)),folder/'projector')
                    row['verification']=verify_measurement(row,folder/'projector');measured.append(row)
                result=dict(variant=variant,task=task,model_sha256=sha(final),frozen=diagnostic,measurements=measured)
                write(folder/'result.json',result);l6rows.append(result)
            except Exception as error:failure(variant,6,task['sample_id'],task['seed'],error)
            write(root/'l6_results.json',dict(rows=l6rows,ground_state_qualified=False))
    status('completed_with_failures_or_blocked_arms' if failures or any(not g['pass_diagnostic_screen'] for g in gates.values())
           else 'completed_unqualified')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['output','l2','l6','regressions']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(args)


if __name__=='__main__':main()
