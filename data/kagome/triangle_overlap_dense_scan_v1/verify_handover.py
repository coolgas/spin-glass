"""Read-only live ownership/resource audit, run on AutoDL on demand."""
import json
from pathlib import Path
import shutil
import time

from transformer_qmc.triangle_correlation_parallel import alive, key, owned_workers
from transformer_qmc.triangle_overlap_dense import external_workers


base = Path('/root/autodl-tmp')
dense = base/'triangle-overlap-dense-scan-20261105-v1'
original = base/'triangle-correlation-retention-20261105-v6/results'
extension = base/'triangle-correlation-highchi-20261105-v7/results'
handover = next(dense/name for name in ('handover-v5','handover-v4','handover-v3') if (dense/name/'launch.json').exists())
launch = json.loads((handover/'launch.json').read_text())
a = json.loads((original/'status.json').read_text())
b = json.loads((dense/'results/status.json').read_text())
original_workers = owned_workers(original)
assert {j['pid'] for j in a['running']} == {j['pid'] for j in original_workers}
assert all(alive(launch[name]['pid']) for name in ('original','dense'))
assert all(alive(pid) for pid in launch['preserved_workers'])
assert not alive(launch['held_controller'])
assert json.loads((extension/'status.json').read_text())['state']=='held_for_requested_dense_campaign'
workers = external_workers([original])+[j for j in b['running'].values() if alive(j['pid'])]
owners = [(key(j['task']), j['pid']) for j in workers if j.get('mode','case')=='case']
assert len(owners) == len({name for name,pid in owners}), owners
threads = sum(j['threads'] for j in workers)
memory = sum(j['memory_gib'] for j in workers)
assert threads <= 16 and memory <= 72
extra = 0.
for j in workers:
    try:
        lines = Path(f"/proc/{j['pid']}/status").read_text().splitlines()
        rss = next(int(line.split()[1])/2**20 for line in lines if line.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        rss = 0.
    extra += max(0.,j['memory_gib']-rss)
cg = Path('/sys/fs/cgroup')
memory_stat = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
available = (int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text()))/2**30
reclaimable = min(int(memory_stat['inactive_file']), int(memory_stat['file'])-int(memory_stat['shmem']))/2**30
evidence = dict(utc=time.time(), launch=launch,
    assertions=dict(controllers_live=True, original_workers_adopted=True,
                    preserved_workers_live=True, extension_held=True,
                    unique_case_owners=True, shared_budgets_obeyed=True),
    original_status=a, dense_status=b, workers=workers,
    shared_threads=threads, shared_memory_reservations_gib=memory,
    memory_unused_gib=available, memory_available_gib=available+max(0.,reclaimable),
    memory_reclaimable_inactive_file_gib=max(0.,reclaimable), existing_worker_growth_gib=extra,
    first_validation_admission_need_gib=16+extra+4,
    cgroup_memory_anon_gib=int(memory_stat['anon'])/2**30,
    cgroup_memory_file_gib=int(memory_stat['file'])/2**30,
    disk_free_gib=shutil.disk_usage(base).free/2**30,
    preserved_other_processes={str(pid):alive(pid) for pid in (515820,335206)})
path = handover/'verification.json'
path.write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({k:v for k,v in evidence.items() if k not in ('original_status','dense_status','workers','launch')},indent=2))
