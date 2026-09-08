#!/usr/bin/env bash
# Submit 5-seed sweeps for the 2×2 contextual-MARL matrix (N=5, randdrop).
# Override MATRIX=static_baseline,varying_baseline,... to run a subset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

TOTAL_STEPS="${TOTAL_STEPS:-200000}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-1 2 3 4 5})
MATRIX="${MATRIX:-static_baseline,varying_baseline,static_context,varying_context}"

submit_baseline() {
  local regime="$1"
  for seed in "${SEEDS[@]}"; do
    job_id="$(
      sbatch --parsable \
        --export=ALL,SEED="$seed",TOTAL_STEPS="$TOTAL_STEPS",TRAINING_REGIME="$regime" \
        cluster/slurm_baseline5.sh
    )"
    echo "  baseline regime=${regime} seed=${seed} -> job ${job_id}"
  done
}

submit_context() {
  local regime="$1"
  for seed in "${SEEDS[@]}"; do
    job_id="$(
      sbatch --parsable \
        --export=ALL,SEED="$seed",TOTAL_STEPS="$TOTAL_STEPS",TRAINING_REGIME="$regime",CONTEXT_MODE=concat \
        cluster/slurm_context5_concat.sh
    )"
    echo "  context concat regime=${regime} seed=${seed} -> job ${job_id}"
  done
}

IFS=',' read -ra entries <<< "$MATRIX"
for entry in "${entries[@]}"; do
  case "$entry" in
    static_baseline) echo "Submitting static hidden-context baseline..."; submit_baseline static ;;
    varying_baseline) echo "Submitting varying hidden-context baseline..."; submit_baseline varying ;;
    static_context) echo "Submitting static context-concat..."; submit_context static ;;
    varying_context) echo "Submitting varying context-concat..."; submit_context varying ;;
    *) echo "Unknown matrix entry: $entry" >&2; exit 1 ;;
  esac
done

echo
echo "Note: varying regime updates physics in-place (no env recreate; 32G is enough)."
echo "Monitor:  squeue -u \$USER"
echo "Aggregate examples:"
echo "  python scripts/rliable_aggregate.py --preset baseline5_varying"
echo "  python scripts/rliable_aggregate.py --preset context5_varying --compare-preset baseline5_varying"
echo "Notebook: notebooks/evaluate_rliable.ipynb (MODE=compare)"
