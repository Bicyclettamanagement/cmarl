#!/usr/bin/env bash
# Example SLURM job for the context-free MATD3 Pistonball baseline.
# Adjust partition, GPU type, and time limit for your cluster.
#SBATCH --job-name=cmarl-matd3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cmarl

SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python matd3_pistonball.py \
  --seed "$SEED" \
  --total-timesteps "$TOTAL_STEPS" \
  --checkpoint-eval \
  --transfer-eval

echo "End: $(date -Is)"
