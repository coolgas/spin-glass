"""Frozen user-selected chi12/D128 campaign, 128 matched disorder geometries."""
from .config import drive_grid


def tasks():
    rows=[]
    for sid in range(128):
        for index,g in enumerate(drive_grid()):
            for delta in (2.,3.15):
                for model in ('full','nn_nnn','nn'):
                    rows.append(dict(id=f's{sid:03d}_{model}_d{delta:g}_g{index:02d}_c12_D128',
                        sample=sid,model=model,delta=delta,omega=float(g),chi=12,
                        dimensions=[128],pairs=20000,fixed_production=True,
                        no_exact=not (sid==0 and index in (0,14,29))))
    return rows
