# Rydberg kagome simulations

- Shared project root: `/mnt/ali/yty/spin-glasses/rydberg-kagome`, covering L4,
  L6 and future L8. Use lattice sizes in campaign names. Preserve the former
  `rydberg-kagome-l4-pq` compatibility symlink for running jobs, Git worktree
  metadata and historical provenance.

- `code/rydberg_kagome` is the byte-verified production v5 source. Preserve the
  tagged baseline; make future changes on a named branch with a new release ID.
- Keep the notebook filename `notebook/rydberg_kagome_L4_Pq_progress.ipynb`.
- Store datasets, environments, caches and outputs alongside `repository/`, not
  inside it. Require explicit output paths for computations.
- Retain all 128 disorders and six numerical flags. Do not claim RG convergence.
- Scientific plots require LaTeX typography, closed panel frames and the original
  definitions, disorder weighting and bootstrap seeds.
- `provenance/notebook-history` preserves original machine-specific editing
  scripts; use `scripts/` for portable entry points.
- Check hashes with `scripts/verify_snapshot.py` after transfer. No production
  campaign is authorized by deploying or validating this repository.
