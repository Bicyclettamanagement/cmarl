#!/usr/bin/env bash
# Bootstrap the reproducible conda environment on a Linux cluster node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-cmarl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load your cluster's miniconda/anaconda module first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists. Updating from environment.yml..."
  conda env update -f environment.yml --prune
else
  echo "Creating environment '$ENV_NAME' from environment.yml..."
  conda env create -f environment.yml
fi

conda activate "$ENV_NAME"

echo "=== Environment snapshot ==="
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import pettingzoo, supersuit; print('pettingzoo', pettingzoo.__version__)"
python -m pytest tests/test_matd3_pistonball.py -q

conda list --explicit > "conda-${ENV_NAME}-linux.lock"
echo "Wrote pinned lock file: conda-${ENV_NAME}-linux.lock"
