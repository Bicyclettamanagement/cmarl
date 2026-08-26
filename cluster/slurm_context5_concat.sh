#!/usr/bin/env bash
# N=5 context-concat MATD3 (matches baseline5 HPs for fair comparison).
# Feeds explicit physics context to actor+critic via feature concatenation.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-ctx5-concat
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
EXPERIMENT="${EXPERIMENT:-context5_concat_randdrop}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python matd3_pistonball_context.py \
  --experiment "$EXPERIMENT" \
  --seed "$SEED" \
  --n-pistons 5 \
  --total-timesteps "$TOTAL_STEPS" \
  --buffer-size 100000 \
  --learning-starts 10000 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --actor-lr 1e-4 \
  --policy-frequency 4 \
  --exploration-noise 0.2 \
  --exploration-noise-end 0.05 \
  --exploration-noise-decay-steps 100000 \
  --max-grad-norm 10.0 \
  --share-actors \
  --critic-shared-obs \
  --context-mode concat \
  --context-to-actor \
  --context-to-critic \
  --random-drop \
  --random-rotate \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --transfer-eval \
  --method-tag context_concat

echo "End: $(date -Is)"
echo "Compare against baseline5_shared_randdrop (method_tag=hidden_context)."
