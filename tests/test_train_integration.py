"""End-to-end checks that training emits cluster-ready logs and evaluation artifacts."""

import json

import pytest

from matd3_pistonball import Args, train


@pytest.fixture
def tiny_train_args() -> Args:
    return Args(
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        total_timesteps=220,
        learning_starts=80,
        buffer_size=200,
        batch_size=16,
        eval_frequency=100,
        eval_episodes=2,
        checkpoint_eval=True,
        transfer_eval=True,
        performance_threshold=-500.0,
        cuda=False,
        seed=42,
    )


def test_train_emits_cluster_artifacts(tmp_path, tiny_train_args, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_name = train(tiny_train_args)
    run_dir = tmp_path / "runs" / run_name

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "training_episodes.jsonl").exists()
    assert (run_dir / "eval_history.jsonl").exists()
    assert (run_dir / "rliable_scores.json").exists()
    assert (run_dir / "transfer_results.json").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["finished"] is True
    assert manifest["environment"]["packages"]["pettingzoo"] != "unknown"
    assert manifest["steps_to_threshold"] is not None
    assert manifest["best_eval"] is not None
    assert manifest["best_eval"]["stats"] is not None

    rliable = json.loads((run_dir / "rliable_scores.json").read_text())
    assert rliable["method_tag"] == "hidden_context"
    assert "train" in rliable["context_order"]
    assert len(rliable["context_order"]) > 1

    episodes = (run_dir / "training_episodes.jsonl").read_text().strip().splitlines()
    assert len(episodes) >= 1
    rec = json.loads(episodes[0])
    assert "team_episodic_return" in rec
    assert "per_agent_return_variance" in rec


def test_train_rejects_learning_starts_ge_buffer_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = Args(
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        total_timesteps=50,
        learning_starts=200,
        buffer_size=100,
        batch_size=16,
        eval_frequency=0,
        checkpoint_eval=False,
        transfer_eval=False,
        cuda=False,
        seed=0,
    )
    with pytest.raises(ValueError, match="learning_starts"):
        train(bad)
