#!/usr/bin/env bash
# Multi-seed N=10 context-free baseline (after one successful baseline10 single-seed run).
# Default: 3 seeds to limit queue karma; override with SEEDS="1 2 3 4 5".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-baseline10_shared_randdrop}"
TOTAL_STEPS="${TOTAL_STEPS:-400000}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-1 2 3})

echo "Submitting ${#SEEDS[@]} N=10 baseline jobs (experiment=${EXPERIMENT}, steps=${TOTAL_STEPS})"
for seed in "${SEEDS[@]}"; do
  job_id="$(
    sbatch --parsable \
      --export=ALL,SEED="$seed",TOTAL_STEPS="$TOTAL_STEPS",EXPERIMENT="$EXPERIMENT" \
      cluster/slurm_baseline10.sh
  )"
  echo "  seed=${seed} -> job ${job_id}"
done

echo
echo "Monitor:  squeue -u \$USER"
echo "Runs:     ls -d runs/${EXPERIMENT}__*"
echo "Aggregate when done:"
echo "  python scripts/rliable_aggregate.py --exp-name matd3_pistonball --seeds ${SEEDS[*]}"
