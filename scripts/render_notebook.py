"""Run the exact notebook and export its figures with LaTeX typography."""
import argparse
import json
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.expanduser().resolve()
repo = Path(__file__).resolve().parents[1]
if out.is_relative_to(repo):
    parser.error('Place generated figures outside the source repository')
if not shutil.which('latex') or not shutil.which('dvipng'):
    parser.error('LaTeX and dvipng are required; install TeX including amsmath, type1cm and cm-super. Typography will not be silently changed.')
out.mkdir(parents=True, exist_ok=False)
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib-cache')
import matplotlib.pyplot as plt

count = 0
def save_figures(*args, **kwargs):
    global count
    for number in plt.get_fignums():
        count += 1
        fig = plt.figure(number)
        fig.savefig(out / f'figure-{count:02d}.pdf', bbox_inches='tight')
        fig.savefig(out / f'figure-{count:02d}.png', dpi=150, bbox_inches='tight')
    plt.close('all')

plt.show = save_figures
notebook = json.loads((repo / 'notebook/rydberg_kagome_L4_Pq_progress.ipynb').read_text())
namespace = {'__name__': '__main__'}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)
print(f'Exported {count} figures to {out}')
