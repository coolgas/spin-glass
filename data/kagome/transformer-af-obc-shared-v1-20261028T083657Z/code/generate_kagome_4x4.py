"""Draw the open 4 x 4 Kagome patch.

``build_kagome`` lives in ``kagome_qea.lattice`` and is re-exported here so
``from generate_kagome_4x4 import build_kagome`` keeps working.
"""

from pathlib import Path

import numpy as np

from kagome_qea.lattice import build_kagome

__all__ = ["build_kagome"]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    size = 4
    points, edges, triangles, adjacency = build_kagome(size)
    memberships = np.zeros(len(points), dtype=int)
    for triangle in triangles:
        memberships[list(triangle)] += 1
    assert len(points) == 48
    assert len(edges) == 81
    assert len(triangles) == 25
    assert np.all(memberships >= 1), "A site lacks an intact triangle"
    assert np.all(memberships <= 2)
    # Verify connectedness of the nearest-neighbor graph.
    visited, pending = {0}, [0]
    while pending:
        i = pending.pop()
        for j in np.flatnonzero(adjacency[i]):
            if int(j) not in visited:
                visited.add(int(j))
                pending.append(int(j))
    assert len(visited) == len(points)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(10, 7.4), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.12, top=0.86)
    ax.add_collection(PolyCollection(
        [points[list(t)] for t in triangles],
        facecolors="#F7DCB1", edgecolors="none", zorder=1,
    ))
    ax.add_collection(LineCollection(
        points[edges], colors="#263647", linewidths=1.7, zorder=2,
    ))
    ax.scatter(points[:, 0], points[:, 1], s=53, c="#182C3D",
               edgecolors="white", linewidths=0.65, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(points[:, 0].min() - 0.48, points[:, 0].max() + 0.48)
    ax.set_ylim(points[:, 1].min() - 0.35, points[:, 1].max() + 0.35)
    ax.axis("off")
    fig.text(0.5, 0.958, "4 × 4 kagome lattice", ha="center",
             va="top", fontsize=23, weight="bold", color="#182C3D")
    fig.text(0.5, 0.903, "16 three-site unit cells · 48 sites · open boundaries",
             ha="center", va="top", fontsize=11.5, color="#586675")
    fig.text(0.5, 0.062,
             "Every boundary site belongs to at least one shaded triangle.",
             ha="center", va="center", fontsize=11.5, color="#364859")
    output_dir = Path(__file__).resolve().parents[1]
    for extension in ("png", "svg", "pdf"):
        target = output_dir / f"kagome_4x4.{extension}"
        fig.savefig(target, dpi=240, facecolor="white")
        print(f"Saved {target}")
    plt.close(fig)
    counts = {int(n): int(np.count_nonzero(memberships == n))
              for n in np.unique(memberships)}
    print(f"Verified: {len(points)} sites, {len(edges)} bonds, "
          f"{len(triangles)} intact triangles, connected graph.")
    print(f"Triangle memberships per site: {counts}")


if __name__ == "__main__":
    main()
