"""Atomic result records and content hashes."""
import hashlib
import json
import os
import pickle
import gzip
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def source_hashes():
    root=Path(__file__).parent
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.json')}


def json_write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


class _CheckpointPickler(pickle.Pickler):
    def __init__(self,stream,sites):
        super().__init__(stream,protocol=5)
        self.site_ids={id(site):i for i,site in enumerate(sites)}

    def persistent_id(self,obj):
        index=self.site_ids.get(id(obj))
        return ('mpo-site-v1',index) if index is not None else None


class _CheckpointUnpickler(pickle.Unpickler):
    def __init__(self,stream,sites):
        super().__init__(stream)
        self.sites=sites

    def persistent_load(self,reference):
        tag,index=reference
        if tag!='mpo-site-v1' or not 0<=index<len(self.sites):
            raise pickle.UnpicklingError('Checkpoint requires matching reconstructed MPO sites')
        return self.sites[index]


def checkpoint(path,value,sites=()):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(tmp,'wb') as f: _CheckpointPickler(f,sites).dump(value)
    with opener(tmp,'rb') as f: _CheckpointUnpickler(f,sites).load()
    os.replace(tmp,path)
    return digest(path)


def load_checkpoint(path,sites=()):
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as f: return _CheckpointUnpickler(f,sites).load()


def array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()
