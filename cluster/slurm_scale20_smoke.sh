#!/usr/bin/env bash
# Single-seed smoke: scale shared-actor MATD3 from 5 -> 20 pistons (fixed drop).
# Run this once before the multi-seed baseline sweep.
# Resources sized for ~40k-transition buffer (~34 GB host) + headroom; 4 CPUs enough for 1 env.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-scale20-smoke
#SBATCH --time=10:00:00
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
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
EXPERIMENT="${EXPERIMENT:-scale20_shared_nodrop}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Keep nodrop to stress coordination at N=20 without spawn luck. Transfer off (smoke only).
python matd3_pistonball.py \
  --experiment "$EXPERIMENT" \
  --seed "$SEED" \
  --n-pistons 20 \
  --total-timesteps "$TOTAL_STEPS" \
  --buffer-size 40000 \
  --learning-starts 10000 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --actor-lr 1e-4 \
  --policy-frequency 4 \
  --exploration-noise 0.2 \
  --exploration-noise-end 0.05 \
  --exploration-noise-decay-steps 150000 \
  --max-grad-norm 10.0 \
  --share-actors \
  --critic-shared-obs \
  --no-random-drop \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --no-transfer-eval

echo "End: $(date -Is)"
echo "Go / no-go for full baseline:"
echo "  - eval success should rise well above 0 and not freeze identical across checkpoints"
echo "  - then submit: sbatch cluster/submit_baseline_seeds.sh"
