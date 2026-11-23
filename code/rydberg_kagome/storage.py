"""Bound new tensor writes while preserving the last committed restart pair."""
import os
from pathlib import Path


GIB=2**30
CASE_LIMIT=2*GIB
FILE_LIMIT=256*2**20
LOG_LIMIT=256*2**10


def allocated_bytes(root):
    return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file())


class BoundedWriter:
    def __init__(self,stream,limit):
        self.stream=stream; self.limit=limit; self.written=0

    def write(self,data):
        if self.written+len(data)>self.limit:
            raise OSError('Case storage reservation exhausted; previous restart preserved')
        size=self.stream.write(data); self.written+=size
        return size

    def flush(self): return self.stream.flush()


def checkpoint_limit(path):
    root=os.environ.get('RYDBERG_CASE_ROOT')
    if not root: return None
    if not Path(path).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Checkpoint outside reserved case')
    # Leave space for metadata, final moments, signed histogram and failure record.
    return max(0,min(FILE_LIMIT,CASE_LIMIT-allocated_bytes(root)-16*2**20))


class BoundedLog:
    """Keep a bounded initial log; durable failure records carry terminal errors."""
    def __init__(self,path):
        self.stream=Path(path).open('ab',buffering=0)
        self.remaining=max(0,LOG_LIMIT-Path(path).stat().st_size)
        self.encoding='utf-8'

    def write(self,text):
        data=text.encode('utf-8',errors='replace')
        if self.remaining:
            part=data[:self.remaining]; self.stream.write(part); self.remaining-=len(part)
        return len(text)

    def flush(self): self.stream.flush()

    def isatty(self): return False

    def close(self): self.stream.close()
