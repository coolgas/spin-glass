"""Replace failed low-field plot points only with a validated remedy ensemble."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_correlation_lengths import paired_summary
from .assess_correlation_remedy import assess as assess_development
from .validate_correlation_remedy import assess as assess_heldout
from .correlation_lengths import ROOT, sha
from .extend_correlation_fields import stem


def corrected_comparison(original):
    root=ROOT/'data/kagome/ltfim_correlation_remedy_v1'
    result=original.copy()
    result['method']='Archived neural guide; joint replicas'
    if not (root/'results/manifest.json').exists() or not (root/'heldout/manifest.json').exists():
        return result,None,None
    development=assess_development(root/'results')
    heldout=assess_heldout(root/'heldout')
    if not all(r['passes_development_checks'] for r in development['decisions']) or not heldout['all_checks_pass']:
        raise RuntimeError('Low-field remedy has not passed the required validation')
    # One uniform, longer-projection setting for all three fields. Not a per-field
    # choice of whichever population/repeat happens to agree best with exact.
    selected='time'
    before_after=[]
    for gamma in [.25,.5,.75]:
        raw=[];exact=[];alarms=0
        for sid in range(50000,50008):
            guide=json.loads((root/'results/guides'/(stem(gamma,sid)+'.json')).read_text())
            exact.append(guide['exact_spectrum']);repeats=[]
            for repeat in range(4):
                r=json.loads((root/'results/runs'/(stem(gamma,sid)+f'_{selected}_r{repeat}.json')).read_text())
                repeats.append(r['result']['connected_spectrum']);alarms+=r['ancestry_alarm']
            raw.append(repeats)
        summary=paired_summary(np.asarray(exact),np.asarray(raw))
        index=result.index[result.gamma==gamma][0]
        np.testing.assert_allclose(summary['exact'],result.at[index,'exact'],atol=1e-11)
        old=result.loc[index].copy()
        for key,value in summary.items():result.at[index,key]=value
        result.at[index,'ancestry_alarms']=alarms
        result.at[index,'comparison_status']='passes L2 numerical checks'
        result.at[index,'method']='Corrected neural guide; independent replicas'
        before_after.append(dict(gamma=gamma,original_gfmc=float(old.gfmc),corrected_gfmc=summary['gfmc'],
            matched_exact=summary['exact'],corrected_disorder_mc_se=summary['gfmc_se'],
            original_alarms=int(old.ancestry_alarms),corrected_alarms=alarms))
    metadata=dict(selected_setting=selected,walkers_per_replica=1024,physical_replicas=4,
        equilibration_time=32.,forward_time=32.,branch_interval=.1,cycles=16,repeats=4,disorders=8,
        guide='Frozen neural network multiplied by variational degree-four log-amplitude correction',
        guide_training='Per-context Hamiltonian energy minimization; no exact-state labels',
        exact_labels_used=False,development_runs=development['completed_runs'],heldout_runs=heldout['completed_runs'],
        corrected_test_runs=384+192,corrected_test_ancestry_alarms=sum(r['ancestry_alarms'] for r in development['rows'])+sum(r['ancestry_alarms'] for r in heldout['rows']),
        heldout_ids=heldout['sample_ids'],development_passes=True,heldout_passes=True,
        qualification='Validated at L2 under stated tolerances; enumeration-based guide optimization is not a demonstrated large-L solution.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            root/'results/validated_review.json',root/'heldout/validated_review.json',
            Path(__file__),ROOT/'data/kagome/ltfim_correlation_lengths_v2/l2_comparison.csv']})
    result.to_csv(root/'corrected_comparison.csv',index=False)
    pd.DataFrame(before_after).to_csv(root/'before_after.csv',index=False)
    (root/'selection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return result,pd.DataFrame(before_after),metadata
