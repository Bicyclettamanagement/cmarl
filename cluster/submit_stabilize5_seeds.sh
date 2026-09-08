#!/usr/bin/env bash
# Multi-seed N=5 stabilize/randdrop sweep (cheap context-free reference).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-stabilize5_shared_randdrop}"
TOTAL_STEPS="${TOTAL_STEPS:-300000}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-1 2 3})

echo "Submitting ${#SEEDS[@]} N=5 stabilize jobs (experiment=${EXPERIMENT}, steps=${TOTAL_STEPS})"
for seed in "${SEEDS[@]}"; do
  job_id="$(
    sbatch --parsable \
      --export=ALL,SEED="$seed",TOTAL_STEPS="$TOTAL_STEPS",EXPERIMENT="$EXPERIMENT" \
      cluster/slurm_stabilize5_randdrop.sh
  )"
  echo "  seed=${seed} -> job ${job_id}"
done

echo
echo "Monitor:  squeue -u \$USER"
echo "Runs:     ls -d runs/${EXPERIMENT}__*"
