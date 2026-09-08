#!/usr/bin/env bash
# Back-compat: N=20 sweep is paused until N=10 works. Prefer submit_baseline10_seeds.sh.
set -euo pipefail
echo "NOTE: prefer cluster/submit_baseline10_seeds.sh (N=20 smoke failed)."
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_baseline10_seeds.sh" "$@"
