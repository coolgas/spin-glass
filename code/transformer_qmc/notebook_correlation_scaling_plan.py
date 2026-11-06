"""Build a local planning notebook; only dimension arithmetic, no simulations."""
import hashlib
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name('CORRELATION_LENGTH_SCALING_OUTLINE.md')
OUTPUT = ROOT / 'notebook/triangle_rg_correlation_length_scaling_plan.ipynb'


def memory(size):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB'):
        if size < 1024 or unit == 'EiB':
            return f'{size:,.2f} {unit}'
        size /= 1024


def resource_table():
    rows = ['| L | N | chi | Top blocks | D | One vector | 32 vectors | Current N-vector measurement payload |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for L, chi in ((4, 3), (4, 16), (6, 3), (6, 16), (8, 3), (8, 16)):
        n, blocks = 3*L*L, (L//2)**2
        dimension = chi**blocks
        rows.append(f'| {L} | {n} | {chi} | {blocks} | {dimension:,} | '
                    f'{memory(8*dimension)} | {memory(32*8*dimension)} | '
                    f'{memory(n*8*dimension)} |')
    return '\n'.join(rows)


def main():
    text = PLAN.read_text()
    expanded = text.replace('{RESOURCE_TABLE}', resource_table())
    sections = expanded.split('\n## ')
    cells = [nbformat.v4.new_markdown_cell(sections[0])]
    cells.extend(nbformat.v4.new_markdown_cell('## '+section) for section in sections[1:])
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'plan_only': True, 'scientific_runs_launched': False,
        'planned_compute_location': 'AutoDL',
        'source_plan': str(PLAN.relative_to(ROOT)),
        'source_plan_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'error_bar_policy': 'disorder only',
        'observable': 'finite-size qEA; connected and raw equal-time chiSG; raw replica q2',
    })
    nbformat.validate(notebook)
    OUTPUT.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()
