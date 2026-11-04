"""Export source snapshots and run specifications from a local result archive.

Run from the full archive checkout; the destination must be another checkout.
Large samples, model checkpoints, and measured result tables are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    destination = args.destination.resolve()
    if destination == root:
        parser.error('destination must differ from the source checkout')
    names = subprocess.check_output(['git', 'ls-files', '-z', 'data'], cwd=root).decode().split('\0')
    records = []
    for name in sorted(filter(None, names)):
        source = root / name
        if source.is_symlink():
            continue
        suffix = source.suffix.lower()
        metadata = suffix == '.json' and any(word in source.stem.lower() for word in
            ('manifest', 'launch', 'config', 'environment', 'protocol', 'tasks', 'provenance'))
        if suffix not in ('.py', '.sh', '.md', '.toml', '.yaml', '.yml', '.ini') and not metadata:
            continue
        target = destination / name
        contents = source.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        elif target.read_bytes() != contents:
            raise RuntimeError(f'Destination differs: {name}')
        records.append({'path': name, 'sha256': hashlib.sha256(contents).hexdigest(),
                        'bytes': len(contents)})
    output = destination / 'reproduction/source_inventory.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'files': records}, indent=2) + '\n')
    print(f'Preserved {len(records)} source/configuration files ({sum(r["bytes"] for r in records)/2**20:.1f} MiB)')


if __name__ == '__main__':
    main()
