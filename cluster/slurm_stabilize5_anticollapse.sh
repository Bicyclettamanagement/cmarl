#!/usr/bin/env bash
# Next experiment after failed N=10 / N=20 scale smokes.
#
# Diagnosis: N=10 nodrop never got a success signal (0% success, all len=125 for 400k
# steps). That is a coordination/credit-assignment wall, not a near-miss on LR/batch.
# Retuning N=10 alone is low EV; first recover a stable N=5 randdrop policy.
#
# Builds on bridge5 (learned to ~98 then late-collapsed via action saturation) and
# avoids stabilize5's failure mode (random_rotate + immediate sat=1 deterministic
# policy / frozen eval). Anti-collapse knobs: slower actor updates, lower actor LR,
# higher late exploration floor, no random_rotate yet.
#
# Resources: N=5 + 100k buffer ~21 GB host → 32G / 1 GPU / 4 CPUs is enough.
# 300k steps took ~1.7h previously → 3h wall is comfortable.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-stab5-anticollapse
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
TOTAL_STEPS="${TOTAL_STEPS:-300000}"
EXPERIMENT="${EXPERIMENT:-stabilize5_anticollapse_randdrop}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python matd3_pistonball.py \
  --experiment "$EXPERIMENT" \
  --seed "$SEED" \
  --n-pistons 5 \
  --total-timesteps "$TOTAL_STEPS" \
  --buffer-size 100000 \
  --learning-starts 10000 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --actor-lr 5e-5 \
  --policy-frequency 8 \
  --exploration-noise 0.2 \
  --exploration-noise-end 0.1 \
  --exploration-noise-decay-steps 250000 \
  --max-grad-norm 10.0 \
  --share-actors \
  --critic-shared-obs \
  --random-drop \
  --no-random-rotate \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --transfer-eval \
  --method-tag hidden_context

echo "End: $(date -Is)"
echo "Success criteria (judge best_eval, not final step):"
echo "  - peak eval success_rate >= 0.9 and return_mean >= ~90 (bridge5-like)"
echo "  - late eval not stuck at action_saturation_mean ~= 1.0 with action_std ~= 0"
echo "  - eval checkpoints are NOT identical across steps"
echo "If OK: seed-sweep this script; only then reconsider N=7 curriculum or N=10."
echo "If fail: keep N=5, ablate actor_lr / policy_frequency before any N>5 retry."
