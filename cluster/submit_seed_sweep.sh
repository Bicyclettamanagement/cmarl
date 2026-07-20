#!/usr/bin/env bash
# Submit multiple seeds for the pre-study baseline (mean + variance across seeds).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SEEDS=(1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
  sbatch --export=ALL,SEED="$seed" cluster/slurm_baseline.sh
done

echo "Submitted ${#SEEDS[@]} jobs."
echo "After completion, aggregate with:"
echo "  python scripts/rliable_aggregate.py --exp-name matd3_pistonball --seeds ${SEEDS[*]}"
