#!/usr/bin/env bash
# Sanity-check MATD3 on a smaller Pistonball instance (5 independent actors).
# Use this to verify learning before scaling back to the 20-piston baseline.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-matd3-sanity5
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# Some conda activation scripts (e.g. MKL/BLAS) reference optional vars like
# `MKL_INTERFACE_LAYER` while `nounset` (`set -u`) is enabled. Temporarily
# relax nounset during activation to avoid hard failures.
set +u
conda activate cmarl
set -u

SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 5 pistons: ~5x less buffer RAM than the 20-agent default, so 100k transitions fit.
# Disable transfer eval to keep the job focused on whether ID training learns at all.
python matd3_pistonball.py \
  --seed "$SEED" \
  --n-pistons 5 \
  --total-timesteps "$TOTAL_STEPS" \
  --buffer-size 100000 \
  --learning-starts 10000 \
  --batch-size 128 \
  --max-grad-norm 10.0 \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --no-transfer-eval

echo "End: $(date -Is)"
echo "Sanity signals to check:"
echo "  - training_episodes.jsonl: any success=true, rising returns"
echo "  - eval_history.jsonl: changing returns across steps (not identical lists)"
echo "  - TensorBoard: non-zero actor grad on policy updates; Q values not collapsing"
