#!/usr/bin/env bash
# N=5 context-concat MATD3. Set TRAINING_REGIME=static|varying (default: varying).
# Context injection: CONTEXT_MODE=concat|none; CONTEXT_TO_ACTOR/CRITIC=0|1.
# Varying regime mutates physics in-place (no per-episode env recreate) → 32G is enough.
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

export PYTHONUNBUFFERED=1

SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
TRAINING_REGIME="${TRAINING_REGIME:-varying}"
TRAIN_CONTEXT_SAMPLING="${TRAIN_CONTEXT_SAMPLING:-random}"
CONTEXT_MODE="${CONTEXT_MODE:-concat}"
EXPERIMENT="${EXPERIMENT:-context5_${CONTEXT_MODE}_${TRAINING_REGIME}_randdrop}"

EXTRA_ARGS=()
if [[ "${CONTEXT_TO_ACTOR:-1}" == "1" ]]; then EXTRA_ARGS+=(--context-to-actor); fi
if [[ "${CONTEXT_TO_CRITIC:-1}" == "1" ]]; then EXTRA_ARGS+=(--context-to-critic); fi

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
echo "training_regime=${TRAINING_REGIME} sampling=${TRAIN_CONTEXT_SAMPLING} context_mode=${CONTEXT_MODE} experiment=${EXPERIMENT}"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python matd3_pistonball_context.py \
  --experiment "$EXPERIMENT" \
  --seed "$SEED" \
  --n-pistons 5 \
  --total-timesteps "$TOTAL_STEPS" \
  --training-regime "$TRAINING_REGIME" \
  --train-context-sampling "$TRAIN_CONTEXT_SAMPLING" \
  --context-mode "$CONTEXT_MODE" \
  "${EXTRA_ARGS[@]}" \
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
  --random-drop \
  --random-rotate \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --transfer-eval

echo "End: $(date -Is)"
