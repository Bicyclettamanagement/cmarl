"""Canonical on-disk layout for training runs and discovery.

Layout (TensorBoard-friendly)::

    runs/
      pistonball/n5/     # main N=5 matrix
      pistonball/n8/     # N=8 coordination pilots
      multiwalker/       # Multiwalker reward / context pilots
    old_runs/            # archived experiments (not on the default TB path)

Point TensorBoard at a leaf, e.g. ``tensorboard --logdir runs/multiwalker``,
or several with ``--logdir_spec=n5:runs/pistonball/n5,mw:runs/multiwalker``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

DEFAULT_RUNS_ROOT = Path("runs")
RUN_MANIFEST_FILENAME = "run_manifest.json"


def runs_subdir_for_args(args: Any) -> Path:
    """Relative subdirectory under ``runs/`` for a training job."""
    env = str(getattr(args, "env_name", "") or "").lower()
    exp = str(getattr(args, "experiment", "") or getattr(args, "exp_name", "") or "").lower()
    if env == "multiwalker" or exp.startswith("mw") or "multiwalker" in exp:
        return Path("multiwalker")
    n = int(getattr(args, "n_pistons", 5) or 5)
    return Path("pistonball") / f"n{n}"


def resolve_run_dir(args: Any, run_name: str, *, runs_root: Path | str = DEFAULT_RUNS_ROOT) -> Path:
    """Full path ``runs/<family>/.../<run_name>`` for a new training run."""
    return Path(runs_root) / runs_subdir_for_args(args) / run_name


def iter_run_dirs(runs_root: Path | str) -> Iterator[Path]:
    """Yield run directories under ``runs_root`` recursively.

    A directory counts as a run if it contains ``run_manifest.json`` and/or
    ``rliable_scores.json`` (legacy score-only fixtures).
    """
    root = Path(runs_root)
    if not root.is_dir():
        return
    found: set[Path] = set()
    for marker in (RUN_MANIFEST_FILENAME, "rliable_scores.json"):
        for path in root.rglob(marker):
            found.add(path.parent)
    for child in sorted(found, key=lambda p: str(p)):
        yield child
