#!/usr/bin/env bash
# Immediate next step: N=10 shared-actor smoke (fixed drop).
# Go/no-go before randdrop baseline10. Lean resources: 80k buffer ~34 GB host.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-scale10-smoke
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate cmarl
set -u

SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-400000}"
EXPERIMENT="${EXPERIMENT:-scale10_shared_nodrop}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python matd3_pistonball.py \
  --experiment "$EXPERIMENT" \
  --seed "$SEED" \
  --n-pistons 10 \
  --total-timesteps "$TOTAL_STEPS" \
  --buffer-size 80000 \
  --learning-starts 10000 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --actor-lr 1e-4 \
  --policy-frequency 4 \
  --exploration-noise 0.2 \
  --exploration-noise-end 0.05 \
  --exploration-noise-decay-steps 200000 \
  --max-grad-norm 10.0 \
  --share-actors \
  --critic-shared-obs \
  --no-random-drop \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --no-transfer-eval

echo "End: $(date -Is)"
echo "Go criteria: some train/eval successes, rising returns, non-identical eval checkpoints."
echo "If OK: sbatch cluster/slurm_baseline10.sh"
echo "If fail: fall back to cluster/slurm_stabilize5_randdrop.sh + seed sweep."
