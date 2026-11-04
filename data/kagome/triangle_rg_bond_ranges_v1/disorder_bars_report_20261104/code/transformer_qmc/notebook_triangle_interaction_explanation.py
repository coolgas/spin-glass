"""Add model definitions and interpretation to the saved intertriangle report."""
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'data/kagome/triangle_rg_interaction_v1/results'


def main():
    path = RESULTS/'triangle_effective_interaction_scan.ipynb'
    nb = nbformat.read(path,as_version=4)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    cells = [nbformat.v4.new_markdown_cell(r'''# What K and lambda mean in this study

**This notebook studies weakened bonds between triangles.** It is the completed
complementary diagnostic. The separate `triangle_effective_bond_scan.ipynb`
lowers both endpoints of the distribution for **every** bond.

## The sampled interactions

First draw positive antiferromagnetic bonds independently from
$K^{(0)}_{ij}\sim U(0.5,1.5)$, with mean 1. Keep that realization fixed
(disorder60000), then multiply only bonds **between** the selected triangles
by $\lambda$. No new disorder is drawn when lambda changes.

| lambda | K inside selected triangles | K between selected triangles | Mean inside / between |
| ---: | :--- | :--- | :--- |
| 0 | U(0.5, 1.5) | exactly zero | 1 / 0 |
| 0.1 | U(0.5, 1.5) | U(0.05, 0.15) | 1 / 0.1 |
| 0.3 | U(0.5, 1.5) | U(0.15, 0.45) | 1 / 0.3 |
| 1 | U(0.5, 1.5) | U(0.5, 1.5) | 1 / 1 |

The code stores $J=-K$; the physical antiferromagnetic K in this table is
positive. L4 has 48 physical spins. L2 has 12; the seed includes size, so its
disorder60000 bonds differ from those at L4. All fields in this study are
$h=0.15$ and $\Gamma=0.25,0.5,1,2$, in units of the mean original internal K.

## Lambda changes the physical Hamiltonian

Choose one covering of the kagome lattice by disjoint triangles (orientation0).
Call their internal edges A and all remaining nearest-neighbor edges B. Then

$$H(\lambda)=
\sum_{(ij)\in A}K^{(0)}_{ij}Z_iZ_j
+\lambda\sum_{(ij)\in B}K^{(0)}_{ij}Z_iZ_j
-h\sum_i Z_i-\Gamma\sum_i X_i.$$

Each kagome site has two A bonds and two B bonds. A and B each contain half
the nearest-neighbor bonds. A B bond connects two different selected triangles;
it need not cross a later four-triangle superblock boundary.

- **lambda=0:** B bonds vanish. The system becomes independent three-spin
  triangles in the fields. The full ground state is a product across triangles.
- **lambda=0.1:** triangles have mean internal K=1 and mean connecting K=0.1.
  Local triangle physics is strongly coupled, while different triangles interact weakly.
- **lambda=1:** every bond has its original strength. This restores the original
  random kagome Hamiltonian, with the same bond law on both triangle orientations.

Lambda is a **dimensionless ratio of boundary to internal interaction strength**.
Changing lambda changes the physical model. It does not merely change the RG
approximation. For lambda<1 the chosen triangle orientation is physically
stronger than the other orientation. By comparison, m and chi change the
retained approximation for a fixed Hamiltonian.
'''), nbformat.v4.new_markdown_cell(r'''## Why lambda can change P(q) so much

Turning on lambda connects independent frustrated triangles into a lattice.
Connecting bonds can select among competing low-energy triangle configurations,
modify correlations and shift the competition with the transverse field.
For example, at Gamma=0.5 the mean connecting K rises from 0.1 at lambda=0.1
to 1 at lambda=1, while Gamma and the internal bonds stay fixed. Those are
substantially different ratios of interaction to field.

Small **bare** lambda is not enough to guarantee a small response. If triangle
configurations are close in energy, even weak connecting bonds can mix or
reorder them. The relevant perturbative comparison is the matrix element
coupling retained and discarded states divided by a connected excitation
denominator. Small denominators can defeat a weak-bond argument.

There are two reasons curves can move when lambda changes: the physical
ground state changes, and the RG truncation error can also change. The first
is the intended physics; the second is assessed by varying m and chi and
using exact L2 checks. We cannot assign the entire L4 shift to accurate
physical response without a matched full-spin L4 benchmark.

Here the measured overlap is

$$q=\frac1{48}\sum_i z_i^{(a)}z_i^{(b)},\qquad
\langle q\rangle=\frac1{48}\sum_i\langle Z_i\rangle^2,$$

for two independent replicas. Thus the finite longitudinal field creates a
positive overlap background. P(q) can move because local polarization changes,
even at lambda=0 where different triangles have no collective interaction.
A large shift does not by itself identify a glass transition. Weakening
antiferromagnetic bonds need not monotonically increase or decrease overlap
in a frustrated system with both fields.
''')]
    text = '''## How large was the observed lambda response?

These are **RG estimates at (m,chi)=(6,16)** of the physical mean overlap.
They illustrate the saved results; they are not full-spin L4 reference values.

| Gamma | lambda=0 | lambda=0.1 | lambda=0.3 | lambda=1 |
| ---: | ---: | ---: | ---: | ---: |
'''
    for g in (.25,.5,1.,2.):
        means=[]
        for lam in (0.,.1,.3,1.):
            row=json.loads((RESULTS/'cases'/f'l4_s60000_g{g:g}_l{lam:g}_m6_c16.json').read_text())
            means.append(row['metrics']['q'])
        text += f'| {g:g} | '+ ' | '.join(f'{v:.4f}' for v in means)+' |\n'
    text += r'''

At Gamma=0.5 the estimated mean overlap changes from 0.0683 at lambda=0 to
0.7096 at lambda=1. At Gamma=2 all four means are near zero. The effect therefore
depends strongly on the field regime. The large shifts at lower fields require
particular caution because the retention screen fails there.

**What the screen established:** only lambda=0.1 at Gamma=1 and 2 passes the
provisional nonzero-lambda screen. Their field-driven mean-q difference is
-0.00344, with replica sampling SE 0.00239. Larger shifts occur at lower fields,
where chi sensitivity fails. A useful interval combining a strong resolved
shift and stable RG has not yet been established by this study.

## How to read the figures below

- In the main 4-by-4 grid, **rows change Gamma** and **columns change lambda**;
  h is always 0.15. Moving across changes the physical boundary bonds in the table above.
- Within one L4 panel, colors compare retention: blue (4,8), orange (6,8),
  green (6,16). Blue to orange tests m; orange to green tests chi. Here m is
  states kept per triangle, chi is states kept per four-triangle superblock.
- The horizontal axis is physical overlap q. The vertical axis is probability
  mass per discrete bin; with 48 spins the spacing is 1/24. Error bars describe
  independent replica sampling, not truncation uncertainty.
- The black L4 curve exists only at lambda=0, where the product-triangle
  reference is exact. The separate L2 companion has exact references at all lambdas.
- In the final field-overlay figure, each panel fixes lambda and the colors
  change Gamma. This directly displays field-driven motion at a fixed bond law.

The original figures and detailed screening tables follow.
'''
    cells.append(nbformat.v4.new_markdown_cell(text))
    for cell in cells:
        cell.metadata['interaction_model_explanation']=True
    nb.cells=cells+[c for c in nb.cells if not c.metadata.get('interaction_model_explanation')]
    nbformat.validate(nb)
    nbformat.write(nb,path)
    destination=ROOT/'notebook/triangle_effective_interaction_scan.ipynb'
    destination.write_bytes(path.read_bytes())
    provenance=dict(note='Explanatory text added to original saved-data report; scientific results unchanged',
        input_notebook_sha256=original_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (RESULTS/'explanation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    verification=RESULTS/'local_verification.json'
    if verification.exists():
        data=json.loads(verification.read_text())
        data['notebook_sha256']=provenance['notebook_sha256']
        data['explanation_source_sha256']=provenance['source_sha256']
        verification.write_text(json.dumps(data,indent=2)+'\n')
    print(destination)


if __name__=='__main__': main()
