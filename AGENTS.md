# Scientific data and storage safety

This is an ongoing scientific project. Before bulk transfers, checkpoint generation, storage changes, or recovery, read `~/.config/opencode/skills/research-storage-safety/SKILL.md` when available. These rules also apply if that skill is unavailable:

- Stage remote campaign downloads, extracted checkpoints, caches, and active compute outputs outside cloud-synced folders, preferably under a verified local `~/ResearchLocal/spin-glasses/` directory. Do not silently change existing jobs or paths.
- Do not recursively mirror remote runs into this Dropbox repository. Check both file count and bytes; use stable archives/manifests and publish only selected, verified snapshots.
- `.gitignore` does not prevent Dropbox synchronization. Do not add high-churn bulk outputs to Git as an automatic step.
- Preserve configurations, seeds, code revisions, resume checkpoints, and transfer provenance. Verify data completeness before deleting any source. Preserve incomplete `.download`/`.part` files until their provenance is resolved.
- Do not relocate this live repository, prune Git objects, reset cloud-provider databases, or delete existing research data as incidental cleanup.
- Follow the additional notebook-location convention in `data/AGENTS.md`.

On 2026-11-06, macOS Save-dialog hangs coincided with a large checkpoint-file arrival and an overloaded Dropbox File Provider domain. That domain was already permanently disconnected. The causal role of this project was not conclusively established; avoid presenting it as proven.
