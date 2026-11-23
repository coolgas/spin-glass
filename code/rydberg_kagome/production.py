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


def adaptive_l6_tasks():
    """User-selected full-interaction L6 adaptive ensemble (7,680 cases)."""
    rows=[]
    for sid in range(128):
        for index,g in enumerate(drive_grid()):
            for delta in (2.,3.15):
                rows.append(dict(id=f'L6_s{sid:03d}_full_d{delta:g}_g{index:02d}_chi12_adaptive',
                    sample=sid,model='full',delta=delta,omega=float(g),chi=12,
                    dimensions=[128,256,384,512],pairs=20000,max_sweeps=120,
                    adaptive=True,L=6,no_exact=True,memory_gib=64,checkpoint_interval=30,
                    final_state_retention='full' if sid==0 and index in (0,14,29) else 'analysis'))
    return rows
