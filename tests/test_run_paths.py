"""Tests for nested run-path layout helpers."""

from pathlib import Path

from evaluation.run_paths import iter_run_dirs, resolve_run_dir, runs_subdir_for_args
from evaluation.rliable_utils import discover_run_dirs


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_runs_subdir_for_args():
    assert runs_subdir_for_args(_Args(env_name="multiwalker", experiment="mw3_x")) == Path("multiwalker")
    assert runs_subdir_for_args(_Args(n_pistons=5, experiment="baseline5")) == Path("pistonball") / "n5"
    assert runs_subdir_for_args(_Args(n_pistons=8, experiment="baseline8")) == Path("pistonball") / "n8"
    assert resolve_run_dir(_Args(n_pistons=5), "run_a") == Path("runs") / "pistonball" / "n5" / "run_a"


def test_discover_run_dirs_nested(tmp_path):
    nested = tmp_path / "pistonball" / "n5" / "baseline5_varying_randdrop__varying__shared__randdrop__n5__seed2__99"
    nested.mkdir(parents=True)
    (nested / "run_manifest.json").write_text('{"finished": true, "args": {"seed": 2}}')
    (nested / "rliable_scores.json").write_text(
        '{"method_tag": "hidden_context", "seed": 2, "context_order": ["train"], '
        '"scores_by_context": {"train": [1.0]}, "algorithm": "x", "global_step": 1, '
        '"eval_split": "zero_shot_transfer"}'
    )
    found = discover_run_dirs(tmp_path, "baseline5_varying_randdrop")
    assert found == [nested]
    assert list(iter_run_dirs(tmp_path)) == [nested]


def test_discover_run_dirs_sanitized_experiment_tag(tmp_path):
    """Dots in experiment tags become underscores in folder names."""
    run_dir = tmp_path / "mw3_concat_varying_fwd5_tp0_1__concat__varying__shared__n3__seed3__1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        '{"finished": true, "args": {"seed": 3, "experiment": "mw3_concat_varying_fwd5_tp0.1"}}'
    )
    (run_dir / "rliable_scores.json").write_text(
        '{"method_tag": "context_concat", "seed": 3, "context_order": ["train"], '
        '"scores_by_context": {"train": [1.0]}, "algorithm": "x", "global_step": 1, '
        '"eval_split": "zero_shot_transfer"}'
    )
    found = discover_run_dirs(tmp_path, "mw3_concat_varying_fwd5_tp0.1")
    assert found == [run_dir]
