#!/usr/bin/env bash
# Back-compat wrapper: prefer cluster/submit_baseline_seeds.sh
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_baseline_seeds.sh" "$@"
