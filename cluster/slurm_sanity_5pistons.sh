#!/usr/bin/env bash
# Sanity-check MATD3 on 5 pistons with shared actors, fixed drop, and stabler TD3 updates.
#SBATCH --partition=gpu-single
#SBATCH --job-name=cmarl-matd3-sanity5
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
# Some conda activation scripts (e.g. MKL/BLAS) reference optional vars like
# `MKL_INTERFACE_LAYER` while `nounset` (`set -u`) is enabled. Temporarily
# relax nounset during activation to avoid hard failures.
set +u
conda activate cmarl
set -u

SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
EXPERIMENT="${EXPERIMENT:-sanity5_shared_nodrop}"

echo "Node: $(hostname)"
echo "Start: $(date -Is)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Shared actor + agent ID, fixed ball drop (harder / less spawn luck), slower actor updates.
python matd3_pistonball.py \
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
  --no-random-drop \
  --eval-frequency 25000 \
  --eval-episodes 10 \
  --checkpoint-eval \
  --no-transfer-eval

echo "End: $(date -Is)"
echo "Sanity signals to check:"
echo "  - run folder under runs/${EXPERIMENT}__shared__nodrop__n5__seed${SEED}__*"
echo "  - training_episodes.jsonl: successes with longer episode lengths (not only <=10)"
echo "  - eval_history.jsonl: changing returns across steps (not identical lists)"
echo "  - TensorBoard: action_saturation not stuck near 1; actor_grad_norm non-zero; eval_returns_unchanged=0"
