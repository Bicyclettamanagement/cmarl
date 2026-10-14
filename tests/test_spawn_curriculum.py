"""Spawn-curriculum geometry, the live ball hook, and a short training smoke test."""

import json

import numpy as np
import pytest

from evaluation.checkpoint_eval import detect_family, instantiate_args
from evaluation.spawn_curriculum import CurriculumTrainingEnvironment, SpawnCurriculum
from matd3_pistonball import Args as HiddenArgs
from matd3_pistonball_context import Args as ContextArgs
from matd3_pistonball_curriculum import Args as CurriculumArgs
from matd3_pistonball_curriculum import train as train_hidden
from matd3_pistonball_context_curriculum import Args as CurriculumContextArgs
from matd3_pistonball_context_curriculum import train as train_context


def _curriculum(**kwargs) -> SpawnCurriculum:
    params = dict(
        n_pistons=10,
        rng=np.random.default_rng(0),
        start_piston=4,
        min_distance_px=80.0,
        success_threshold=0.6,
        window=4,
        min_episodes=4,
        max_episodes=6,
        easy_mix=0.0,
        anneal_steps=10,
    )
    params.update(kwargs)
    return SpawnCurriculum(**params)


def test_warmup_does_not_advance_and_mastery_walks_right():
    curriculum = _curriculum()
    for step in range(8):
        info = curriculum.note_outcome(success=True, global_step=step, learning_started=False)
        assert info["advanced"] is False
    assert curriculum.right_index == 4

    infos = [
        curriculum.note_outcome(success=True, global_step=100 + i, learning_started=True)
        for i in range(4)
    ]
    assert infos[-1]["advanced"] is True
    assert infos[-1]["forced"] is False
    assert infos[-1]["stage_success"] == pytest.approx(1.0)
    assert curriculum.right_index == 5


def test_cap_forces_advance_without_calling_it_mastery():
    curriculum = _curriculum(success_threshold=1.1, min_episodes=2, max_episodes=3, window=3)
    last = None
    for i in range(3):
        last = curriculum.note_outcome(success=False, global_step=50 + i, learning_started=True)
    assert last is not None
    assert last["advanced"] is True
    assert last["forced"] is True
    assert curriculum.right_index == 5


def test_mastering_the_last_piston_graduates_to_native_spawn():
    curriculum = _curriculum(start_piston=9, min_episodes=1, window=1, success_threshold=0.5, max_episodes=4)
    assert curriculum.right_index == 9
    info = curriculum.note_outcome(success=True, global_step=10, learning_started=True)
    assert info["just_graduated"] is True
    assert curriculum.graduated is True
    # Further episodes must not walk past the last piston.
    later = curriculum.note_outcome(success=True, global_step=11, learning_started=True)
    assert later["advanced"] is False
    assert curriculum.right_index == 9


def test_live_reset_spawns_on_the_goal_side_and_keeps_reward_scale():
    args = CurriculumArgs(
        n_pistons=10,
        seed=1,
        cuda=False,
        training_regime="static",
        random_drop=True,
        curriculum_easy_mix=0.0,
    )
    env = CurriculumTrainingEnvironment(args)
    try:
        env.reset(seed=1)
        raw = env._raw
        native_center = raw.screen_width - raw.wall_width - raw.ball_radius - 30
        spawn_x = float(raw.ball.position[0])
        left, right = env.curriculum.frontier_interval(raw)
        assert left - 1e-3 <= spawn_x <= right + 1e-3
        assert spawn_x < native_center - 40
        # Distance is what the reward divides by; it must stay away from 0/1px.
        assert raw.distance_to_wall_at_game_start >= 60
        assert env.curriculum.last_was_native is False
    finally:
        env.close()


def test_frontier_band_moves_right_and_graduation_restores_native_drop():
    args = CurriculumArgs(
        n_pistons=10,
        seed=2,
        cuda=False,
        training_regime="static",
        curriculum_start_piston=8,
        curriculum_min_episodes=1,
        curriculum_window=1,
        curriculum_max_episodes=2,
        curriculum_success_threshold=0.5,
        curriculum_easy_mix=0.0,
    )
    env = CurriculumTrainingEnvironment(args)
    try:
        env.reset(seed=2)
        raw = env._raw
        first_right = env.curriculum.frontier_interval(raw)[1]
        info = env.curriculum.note_outcome(success=True, global_step=20, learning_started=True)
        assert info["advanced"] is True
        assert env.curriculum.graduated is False
        env.start_episode(seed=3)
        second_right = env.curriculum.frontier_interval(raw)[1]
        assert second_right > first_right + 20

        env.curriculum.note_outcome(success=True, global_step=40, learning_started=True)
        assert env.curriculum.graduated is True
        env.curriculum.easy_mix = 0.0
        env.start_episode(seed=4)
        assert env.curriculum.last_was_native is True
        native_lo = raw.screen_width - raw.wall_width - raw.ball_radius - 60
        native_hi = raw.screen_width - raw.wall_width - raw.ball_radius
        assert native_lo - 1 <= float(raw.ball.position[0]) <= native_hi + 1
    finally:
        env.close()


def test_curriculum_checkpoint_args_still_load_in_the_original_eval_stack():
    hidden = vars(CurriculumArgs(experiment="curriculum10_hidden_static_randdrop", n_pistons=10))
    context = vars(
        CurriculumContextArgs(experiment="curriculum10_concat_static_randdrop", n_pistons=10)
    )
    assert detect_family(hidden) == "pistonball"
    assert detect_family(context) == "pistonball_context"
    restored_h = instantiate_args(HiddenArgs, hidden)
    restored_c = instantiate_args(ContextArgs, context)
    assert restored_h.n_pistons == 10
    assert restored_c.context_mode == "concat"
    assert not hasattr(restored_h, "spawn_curriculum")


def _tiny_args(cls, **overrides):
    params = dict(
        experiment="curr_tiny",
        n_pistons=4,
        max_cycles=6,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        total_timesteps=140,
        learning_starts=40,
        buffer_size=80,
        batch_size=16,
        eval_frequency=80,
        eval_episodes=1,
        checkpoint_eval=True,
        transfer_eval=False,
        performance_threshold=None,
        share_actors=True,
        cuda=False,
        seed=7,
        training_regime="static",
        curriculum_start_piston=2,
        curriculum_min_episodes=2,
        curriculum_window=2,
        curriculum_max_episodes=3,
        curriculum_success_threshold=0.0,
        curriculum_easy_mix=0.0,
    )
    params.update(overrides)
    return cls(**params)


@pytest.mark.parametrize("train_fn, args_cls", [(train_hidden, CurriculumArgs), (train_context, CurriculumContextArgs)])
def test_curriculum_train_smoke(tmp_path, monkeypatch, train_fn, args_cls):
    monkeypatch.chdir(tmp_path)
    run_name = train_fn(_tiny_args(args_cls))
    matches = list((tmp_path / "runs").rglob(run_name))
    assert matches, run_name
    run_dir = matches[0]
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["finished"] is True
    episodes = [
        json.loads(line)
        for line in (run_dir / "training_episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert episodes
    assert episodes[0]["context"]["spawn_x"] is not None
    assert "frontier_index" in episodes[0]["context"]
    assert "curr" in run_name
