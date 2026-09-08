#!/usr/bin/env bash
# Multi-seed N=5 context-free baseline (locked bridge5 recipe + transfer).
# Default: 5 seeds. Override with SEEDS="1 2 3" etc.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-baseline5_shared_randdrop}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-1 2 3 4 5})

echo "Submitting ${#SEEDS[@]} N=5 baseline jobs (experiment=${EXPERIMENT}, steps=${TOTAL_STEPS})"
for seed in "${SEEDS[@]}"; do
  job_id="$(
    sbatch --parsable \
      --export=ALL,SEED="$seed",TOTAL_STEPS="$TOTAL_STEPS",EXPERIMENT="$EXPERIMENT" \
      cluster/slurm_baseline5.sh
  )"
  echo "  seed=${seed} -> job ${job_id}"
done

echo
echo "Monitor:  squeue -u \$USER"
echo "Runs:     ls -d runs/${EXPERIMENT}__*"
echo "Aggregate when done:"
echo "  python scripts/rliable_aggregate.py --preset baseline5 --seeds ${SEEDS[*]}"
echo "Compare vs context (after context sweep finishes):"
echo "  python scripts/rliable_aggregate.py --preset context5 --compare-preset baseline5 --seeds ${SEEDS[*]}"
echo "Notebook: notebooks/evaluate_rliable.ipynb (MODE=compare)"
