#!/usr/bin/env bash
# Multi-seed N=5 context-concat sweep (fair compare vs submit_baseline5_seeds.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-context5_concat_randdrop}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-1 2 3 4 5})

echo "Submitting ${#SEEDS[@]} context-concat jobs (experiment=${EXPERIMENT}, steps=${TOTAL_STEPS})"
for seed in "${SEEDS[@]}"; do
  job_id="$(
    sbatch --parsable \
      --export=ALL,SEED="$seed",TOTAL_STEPS="$TOTAL_STEPS",EXPERIMENT="$EXPERIMENT" \
      cluster/slurm_context5_concat.sh
  )"
  echo "  seed=${seed} -> job ${job_id}"
done

echo
echo "Monitor:  squeue -u \$USER"
echo "Runs:     ls -d runs/${EXPERIMENT}__*"
