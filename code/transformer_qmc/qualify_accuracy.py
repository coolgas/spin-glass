"""Audit completed extended-campaign artifacts without asserting exact convergence."""

import hashlib
import json
from pathlib import Path
import numpy as np

from .experiment import write_json


def main():
    root=Path(__file__).resolve().parents[2]/'data/kagome/transformer_af_alibi_accuracy_v1'
    summary=json.loads((root/'summary.json').read_text())
    assessments=[]
    for run in summary:
        parent=run['reports']['parent']['sampled']
        selected=run['reports'][run['selected_stage']]['sampled']
        delta=(selected['energy']['mean']-parent['energy']['mean'])/run['N']
        se=float(np.hypot(selected['energy']['se'],parent['energy']['se'])/run['N'])
        assessments.append(dict(L=run['L'],seed=run['seed'],selected_stage=run['selected_stage'],
                                energy_change_per_spin=delta,combined_se=se,
                                resolved_improvement=bool(run['selected_stage']!='parent' and delta < -3*se),
                                variance_ratio=selected['local_energy_variance']/parent['local_energy_variance']))
        for stage in ('vmc','polish'):
            report=run['reports'][stage]
            path=Path(report['checkpoint'])
            local=root/path.parent.name/path.name
            assert hashlib.sha256(local.read_bytes()).hexdigest()==report['sha256']
    sensitivity=[]
    for L in (2,4,6,8):
        pair=sorted([r for r in summary if r['L']==L],key=lambda r:r['seed'])
        if len(pair)!=2:
            continue
        for key in ('energy','q2','chi_connected'):
            a,b=[r['reports'][r['selected_stage']]['sampled'][key] for r in pair]
            scale=3*L*L if key=='energy' else 1
            delta=(a['mean']-b['mean'])/scale
            se=float(np.hypot(a['se'],b['se'])/scale)
            sensitivity.append(dict(L=L,observable=key,delta=delta,se=se,
                                    difference_over_se=delta/se,
                                    resolved_seed_sensitivity=bool(abs(delta)>3*se)))
    result=dict(completed_branches=len(summary),assessments=assessments,seed_sensitivity=sensitivity,
                thresholds='3 combined sampling SE used as descriptive flags, not a multiple-test certification',
                exact_ground_state_convergence_established=False,
                projector_started=False)
    write_json(root/'qualification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
