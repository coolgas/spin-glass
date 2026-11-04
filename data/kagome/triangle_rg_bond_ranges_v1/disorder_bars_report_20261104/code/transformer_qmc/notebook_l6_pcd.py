"""Execute a dated, self-contained L6 PCD research notebook on AutoDL."""
import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


METHOD = """# L6 persistent chains and transverse-field curriculum

Finite diagnostic study: two fixed disorders, both seeds, three arms, 24 stints
per context. Four independent 1,024-walker training islands and 16 negative
temperatures are shared. Each arm receives 24,000 likelihood updates and a shared
Gamma=4 VMC400 initial checkpoint. Direct reset vs direct PCD isolates persistence
between stints. Curriculum PCD changes the field history. It receives only six
target-field stints; direct arms receive 24. Propagation costs differ.

The curriculum uses two stints at each of 4, 3, 2, 1.5, 1.2, 1, .85, .75, .65,
then six at .6. The high starting field is heuristic. At every stint, fresh
populations equilibrate before new training samples are collected. PCD carries
the entire negative ladder and RNG, alongside the guide and optimizer.

Independent Gamma=.6 measurements follow stints 21 and 24: four populations of
512 walkers, equilibration 16, forward time 16, eight cycles, two repeats.
Raw tags and descendant counts independently reconstruct P(q), q2, connected
spectra, energy and genealogy. All contexts and ancestry alarms are retained.

## Research basis

- [Tieleman (2008), PCD](https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf): persistent
  chains track an evolving model when parameter changes are sufficiently small.
- [Desjardins et al. (2010)](https://proceedings.mlr.press/v9/desjardins10a.html):
  plain Gibbs chains can miss RBM modes; tempering can improve learning and sampling.
- [Pilati, Inack and Pieri (2019)](https://arxiv.org/abs/1907.00907): fit the mixed
  walker density and use its square root as the next guide. Their ferromagnetic
  chain used 10,000 walkers, 20–50 stints and 50,000–100,000 updates per stint;
  this experiment uses a substantially smaller budget and a different model.
- [Brodoloni and Pilati (2024)](https://arxiv.org/abs/2407.05978): self-learned
  guides support frustrated EA simulations after explicit population/ansatz checks.
  Their zero-longitudinal-field square lattice differs from this kagome model.

PCD and curriculum are hypotheses for this problem. Temperature round trips,
unique samples, better energy or a completed run do not establish equilibrium.
The ideal log-amplitude error-halving argument applies at fixed Hamiltonian
and exact fitting/projection; field changes introduce a moving target.
Population, propagation-time, start and training saturation remain unresolved.
Two repeats provide descriptive comparisons, not a reliable systematic-error bound.
"""


def build(root):
    # Embed a consistent report snapshot so downloaded notebooks remain executable.
    report = json.loads((root/'research_report.json').read_text())
    status_path = root/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    validation = root.parent/'validation/regressions.json'
    tests = json.loads(validation.read_text()) if validation.exists() else {}
    cells = [nbformat.v4.new_markdown_cell(METHOD), nbformat.v4.new_code_cell(
        'import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n'
        f'report = json.loads({json.dumps(json.dumps(report))})\n'
        f'status = json.loads({json.dumps(json.dumps(status))})\n'
        f'validation = json.loads({json.dumps(json.dumps(tests))})\n'
        'print("Snapshot UTC:", report["updated_utc"])\n'
        'print("Status:", status)\n'
        'print("Validation:", {k: validation.get(k) for k in ["passed", "tests", "skipped", "gpu"]})\n'
        'print("Training stints:", len(report["training"]), "/ 288")\n'
        'print("Physical measurements:", sum(len(r["measurements"]) for r in report["measurements"]), "/ 48")'),
        nbformat.v4.new_markdown_cell('## Adaptation diagnostics\nEnergy at different fields is not directly comparable. Training ESS is genealogy, not a convergence certificate.'),
        nbformat.v4.new_code_cell(
        'training = pd.DataFrame(report["training"])\n'
        'if not training.empty:\n'
        '    display(training[["arm", "context", "stint", "gamma", "mixed_energy_per_site", "minimum_training_lineage_ess", "largest_training_lineage", "negative_carried", "rejected_steps"]])\n'
        '    fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n'
        '    for (arm, context), rows in training.groupby(["arm", "context"]):\n'
        '        rows = rows.sort_values("stint")\n'
        '        label = arm+" / "+context\n'
        '        axes[0].plot(rows.stint+1, rows.gamma, label=label)\n'
        '        axes[1].plot(rows.stint+1, rows.minimum_training_lineage_ess)\n'
        '        axes[2].plot(rows.stint+1, rows.largest_training_lineage)\n'
        '    for ax, title in zip(axes, ["Transverse field", "Worst island lineage ESS", "Largest training lineage"]):\n'
        '        ax.set(xlabel="Completed stint", title=title)\n'
        '    axes[1].axhline(8, color="red", ls=":"); axes[1].set_yscale("log")\n'
        '    axes[2].axhline(.5, color="red", ls=":")\n'
        '    axes[0].legend(fontsize=5); fig.tight_layout(); plt.show()\n'
        'else:\n    print("Training has not completed its first stint.")'),
        nbformat.v4.new_markdown_cell('## Independent physical measurements and comparisons\nEvery result includes raw reconstruction verification. Alarmed measurements remain visible.'),
        nbformat.v4.new_code_cell(
        'physical = []\n'
        'for group in report["measurements"]:\n'
        '    for m in group["measurements"]:\n'
        '        physical.append(dict(arm=group["arm"], sample_id=group["task"]["sample_id"], seed=group["task"]["seed"], stints=group["stints"], repeat=m["task"]["repeat"], energy_per_site=m["energy"]/108, q2=m["q2"], minimum_descendant_ess=m["minimum_descendant_ess"], largest_lineage=m["largest_lineage"], ancestry_alarm=m["ancestry_alarm"], reconstructed=m["verification"]["reconstructed"]))\n'
        'display(pd.DataFrame(physical))\n'
        'display(pd.DataFrame(report["comparisons"]))\n'
        'if physical:\n'
        '    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n'
        '    for group in report["measurements"]:\n'
        '        for m in group["measurements"]:\n'
        '            label = f"{group[\'arm\']} s{group[\'task\'][\'sample_id\']} seed{group[\'task\'][\'seed\']} t{group[\'stints\']} r{m[\'task\'][\'repeat\']}"\n'
        '            axes[0].plot(m["q_grid"], m["mass"], label=label, alpha=.6)\n'
        '            axes[1].plot(m["spectrum"], alpha=.6)\n'
        '    axes[0].set(xlabel="q", ylabel="Probability mass", title="Independent overlap distributions")\n'
        '    axes[1].set(xlabel="Recorded mode index", title="Connected spectra")\n'
        '    axes[0].legend(fontsize=5); fig.tight_layout(); plt.show()\n'
        'else:\n    print("Independent target-field measurements are pending.")\n'
        'print(report["limitations"])\n'
        'assert report["ground_state_qualified"] is False')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=dict(
        kernelspec=dict(name='python3', display_name='Python 3', language='python')))
    NotebookClient(notebook, timeout=300, kernel_name='python3',
                   resources={'metadata':{'path':str(root)}}).execute()
    destination = root/'l6_pcd_curriculum.ipynb'
    temporary = destination.with_suffix('.tmp.ipynb')
    nbformat.write(notebook, temporary); temporary.replace(destination)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    destination.with_suffix('.html').write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    build(parser.parse_args().root)
