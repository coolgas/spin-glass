"""Bounded geometry-convention audit; no many-body or cloud-output side effects."""
import time
import numpy as np
from .geometry import sample,lattice,distances,masks
from .results import array_hash


def geometry_audit(L,count=128):
    start=time.monotonic(); base,cell=lattice(L)
    old_d=distances(base,cell,False)
    old_nn=np.isclose(old_d,1,atol=1e-8,rtol=0)
    old_nnn=np.isclose(old_d,np.sqrt(3),atol=1e-8,rtol=0)
    current=masks(L)
    rows=[]
    for sid in range(count):
        new=sample(L,sid)
        # Independently replay legacy acceptance (same proposal seeds).
        for attempt in range(100000):
            rng=np.random.default_rng(np.random.SeedSequence([20261116,L,sid,attempt]))
            points=((base+.125*rng.normal(size=base.shape)) @ np.linalg.inv(cell).T % 1) @ cell.T
            d=distances(points,cell,False)
            if d.min()>=.5: break
        corrected=distances(points,cell)
        finite=np.isfinite(d)
        row=dict(sample_id=sid,attempts_new=new['attempts'],attempts_old=attempt+1,
                 geometry_sha256=array_hash(new['points']),
                 accepted_geometry_changed=bool(not np.array_equal(new['points'],points)),
                 legacy_geometry_passes_true_hardcore=bool(corrected.min()>=.5),
                 max_distance_difference=float(np.max(abs(d[finite]-corrected[finite]))),
                 max_row_sum_difference=float(np.max(abs((d**-6).sum(1)-(corrected**-6).sum(1)))))
        rows.append(row)
    return dict(L=L,count=count,seconds=time.monotonic()-start,
                nn_mask_changes=int(np.sum(old_nn!=current['nn'])),
                nnn_mask_changes=int(np.sum((old_nn|old_nnn)!=current['nn_nnn'])),
                total_proposals=sum(r['attempts_new'] for r in rows),
                changed_geometries=sum(r['accepted_geometry_changed'] for r in rows),rows=rows)
