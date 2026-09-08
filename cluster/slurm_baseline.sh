#!/usr/bin/env bash
# Context-free MATD3 baseline (20 shared actors) for contextual-MARL comparisons.
# Uses random_drop so the train distribution matches the usual Pistonball / transfer setting.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-baseline20
#SBATCH --time=18:00:00
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
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
EXPERIMENT="${EXPERIMENT:-baseline20_shared_randdrop}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

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
  --exploration-noise-decay-steps 200000 \
  --max-grad-norm 10.0 \
  --share-actors \
  --critic-shared-obs \
  --random-drop \
  --random-rotate \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --transfer-eval \
  --method-tag hidden_context

echo "End: $(date -Is)"
