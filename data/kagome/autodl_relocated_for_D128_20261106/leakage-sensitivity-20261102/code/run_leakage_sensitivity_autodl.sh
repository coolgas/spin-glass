#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/leakage-sensitivity-20261102
export PYTHONPATH=/root/autodl-tmp/geometry-qfi-20261031/deps
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PY=/root/miniconda3/bin/python
SOURCE=/root/autodl-tmp/geometry-qfi-20261031/frustration_scan
trap 'echo "FAILED at line $LINENO"; touch results/FAILED' ERR
echo 'Waiting for our matched-inference notebook finisher (PID 209184).'
tail --pid=209184 -f /dev/null
test -f /root/autodl-tmp/matched-inference-20261102/results/NOTEBOOK_COMPLETE
echo 'Starting equal-quota leakage validation and scan.'
$PY -u code/leakage_sensitivity_scan.py --mode validate --source "$SOURCE" --out results
$PY -u code/leakage_sensitivity_scan.py --mode run --source "$SOURCE" --out results --batch 20
mkdir -p review/results review/notebooks
ln -sfn /root/autodl-tmp/leakage-sensitivity-20261102/results review/results/leakage_sensitivity_20261102
$PY -u code/render_leakage_sensitivity.py --folder results --template plan.ipynb --notebook review/notebooks/rydberg_leakage_sensitivity_results.ipynb
echo 'LEAKAGE_NOTEBOOK_COMPLETE'
