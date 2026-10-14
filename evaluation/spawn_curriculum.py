"""Reverse curriculum on the Pistonball ball's horizontal spawn.

Pistonball's goal is the left wall, but ``random_drop`` only jitters the ball by
±30px around the **right** edge. With 10+ pistons that relay is longer than the
N=5 task these runs already solve, the shared reward arrives too late, and TD3
collapses. This curriculum starts the ball over the goal-side pistons (about a
5-piston chain, the size that already learns) and walks the spawn one piston
to the right once that finish is reliable.

The hook rewrites ``x`` inside ``add_ball``, which is before the env records
``distance_to_wall_at_game_start`` and before it draws the first frame. Reward
normalization and the stacked observation therefore match the curriculum spawn.
Evaluation envs are never hooked, so checkpoints still score the native far spawn.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from evaluation.context_sampling import TrainingEnvironment


def legal_ball_x_range(raw: Any) -> tuple[float, float]:
    """Inclusive center-x limits used by Pistonball's own reset clamp."""
    lo = float(raw.wall_width + raw.ball_radius + 1)
    hi = float(raw.screen_width - raw.wall_width - raw.ball_radius)
    if hi < lo:
        hi = lo
    return lo, hi


def piston_slot_x(raw: Any, index: int) -> float:
    """Ball-center x aligned with piston ``index`` (0 = goal / left wall)."""
    index = int(np.clip(index, 0, int(raw.n_pistons) - 1))
    x = float(raw.wall_width + raw.piston_radius + raw.piston_width * index)
    lo, hi = legal_ball_x_range(raw)
    return float(np.clip(x, lo, hi))


def min_center_x(raw: Any, min_distance_px: float) -> float:
    """Leftmost ball center whose travel is at least ``min_distance_px``.

    The per-step reward divides by the initial distance, so a spawn a few pixels
    from the wall makes ±1px worth tens of return and blows up the critic.
    On a court shorter than ``min_distance_px`` (unit tests) the floor is capped
    so the spawn is not pinned to the right wall.
    """
    lo, hi = legal_ball_x_range(raw)
    target = lo - 1.0 + float(min_distance_px)
    cap = lo + 0.45 * (hi - lo)
    return float(min(max(lo, target), cap, hi))


class SpawnCurriculum:
    """Performance-gated rightward expansion of the ball spawn, plus an easy mix."""

    def __init__(
        self,
        *,
        n_pistons: int,
        rng: np.random.Generator,
        start_piston: int = 4,
        min_distance_px: float = 80.0,
        success_threshold: float = 0.6,
        window: int = 16,
        min_episodes: int = 16,
        max_episodes: int = 48,
        easy_mix: float = 0.25,
        anneal_steps: int = 40_000,
    ):
        self.n_pistons = int(n_pistons)
        if self.n_pistons < 2:
            raise ValueError("spawn curriculum needs at least 2 pistons")
        # Index 4 covers pistons 0..4 — the same chain length as the stable N=5 runs.
        self.start_index = int(np.clip(start_piston, 1, self.n_pistons - 1))
        self.right_index = self.start_index
        self.min_distance_px = float(min_distance_px)
        self.success_threshold = float(success_threshold)
        self.window = max(1, int(window))
        self.min_episodes = max(1, int(min_episodes))
        self.max_episodes = max(self.min_episodes, int(max_episodes))
        self.easy_mix = float(np.clip(easy_mix, 0.0, 1.0))
        self.anneal_steps = max(0, int(anneal_steps))
        self.rng = rng
        self.successes: deque[float] = deque(maxlen=self.window)
        self.stage_episodes = 0
        self.graduated = False
        self.graduated_at_step: int | None = None
        self.global_step = 0
        self.last_spawn_x: float | None = None
        self.last_was_native = False

    @classmethod
    def from_args(cls, args: Any) -> SpawnCurriculum:
        seed = int(getattr(args, "seed", 0))
        return cls(
            n_pistons=int(args.n_pistons),
            rng=np.random.default_rng(seed + 17_341),
            start_piston=int(getattr(args, "curriculum_start_piston", 4)),
            min_distance_px=float(getattr(args, "curriculum_min_distance_px", 80.0)),
            success_threshold=float(getattr(args, "curriculum_success_threshold", 0.6)),
            window=int(getattr(args, "curriculum_window", 16)),
            min_episodes=int(getattr(args, "curriculum_min_episodes", 16)),
            max_episodes=int(getattr(args, "curriculum_max_episodes", 48)),
            easy_mix=float(getattr(args, "curriculum_easy_mix", 0.25)),
            anneal_steps=int(getattr(args, "curriculum_anneal_steps", 40_000)),
        )

    def set_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def describe(self) -> str:
        return (
            "spawn curriculum: "
            f"start_piston={self.start_index} (0=goal) "
            f"min_distance={self.min_distance_px:g}px "
            f"advance when rolling success>={self.success_threshold:g} "
            f"over {self.window} episodes (min {self.min_episodes}, cap {self.max_episodes}) "
            f"easy_mix={self.easy_mix:g} anneal={self.anneal_steps} steps"
        )

    def frontier_interval(self, raw: Any, right_index: int | None = None) -> tuple[float, float]:
        """One-piston handoff band ending on ``right_index``."""
        lo, hi = legal_ball_x_range(raw)
        index = self.right_index if right_index is None else int(right_index)
        right = min(hi, piston_slot_x(raw, index))
        left = max(min_center_x(raw, self.min_distance_px), right - float(raw.piston_width))
        left = float(np.clip(left, lo, right))
        return left, float(right)

    def prefix_interval(self, raw: Any) -> tuple[float, float]:
        """Already-solved region: min distance up to the previous frontier piston."""
        lo, hi = legal_ball_x_range(raw)
        prev = max(self.start_index, self.right_index - 1)
        right = min(hi, piston_slot_x(raw, prev))
        left = max(lo, min_center_x(raw, self.min_distance_px))
        if left > right:
            return self.frontier_interval(raw, self.start_index)
        return float(left), float(right)

    def _easy_probability(self) -> float:
        if self.graduated:
            if self.anneal_steps <= 0 or self.graduated_at_step is None:
                return 0.0
            elapsed = max(0, self.global_step - self.graduated_at_step)
            remaining = max(0.0, 1.0 - elapsed / float(self.anneal_steps))
            return self.easy_mix * remaining
        if self.right_index <= self.start_index:
            return 0.0
        return self.easy_mix

    def choose_x(self, raw: Any, native_x: float) -> float:
        """Return the ball center x for this reset. Native x is kept after graduation."""
        lo, hi = legal_ball_x_range(raw)
        native_x = float(np.clip(native_x, lo, hi))
        self.last_was_native = False
        if self.graduated and self.rng.random() >= self._easy_probability():
            self.last_was_native = True
            self.last_spawn_x = native_x
            return native_x
        if self.graduated:
            # Rehearse the solved court, stopping short of the native far edge.
            left = max(lo, min_center_x(raw, self.min_distance_px))
            right = min(hi, piston_slot_x(raw, max(self.start_index, self.n_pistons - 2)))
            if right < left:
                left, right = lo, hi
        elif self.rng.random() < self._easy_probability():
            left, right = self.prefix_interval(raw)
        else:
            left, right = self.frontier_interval(raw)
        chosen = float(left if right <= left else self.rng.uniform(left, right))
        chosen = float(np.clip(chosen, lo, hi))
        self.last_spawn_x = chosen
        return chosen

    def _advance(self, global_step: int) -> None:
        self.successes.clear()
        self.stage_episodes = 0
        if self.right_index >= self.n_pistons - 1:
            self.graduated = True
            self.graduated_at_step = int(global_step)
        else:
            self.right_index += 1

    def note_outcome(self, *, success: bool, global_step: int, learning_started: bool) -> dict:
        """Record one finished episode and maybe move the frontier.

        Episodes before ``learning_starts`` do not count. A random warmup must not
        walk the ball back to the full horizon, and it should fill the buffer with
        the short task instead.
        """
        self.global_step = int(global_step)
        played_index = self.right_index
        was_graduated = self.graduated
        advanced = False
        forced = False
        rate = 0.0
        if learning_started and not self.graduated:
            self.stage_episodes += 1
            self.successes.append(1.0 if success else 0.0)
            rate = float(np.mean(self.successes))
            mastered = (
                self.stage_episodes >= self.min_episodes
                and len(self.successes) >= min(self.window, self.min_episodes)
                and rate >= self.success_threshold
            )
            hit_cap = self.stage_episodes >= self.max_episodes
            if mastered or hit_cap:
                self._advance(global_step)
                advanced = True
                forced = hit_cap and not mastered
        elif self.successes:
            rate = float(np.mean(self.successes))
        return {
            "frontier_index": int(played_index),
            "frontier_pistons": int(played_index + 1),
            "spawn_x": None if self.last_spawn_x is None else float(self.last_spawn_x),
            "native_spawn": bool(self.last_was_native),
            "stage_success": float(rate),
            "graduated": bool(self.graduated),
            "advanced": bool(advanced),
            "forced": bool(advanced and forced),
            "just_graduated": bool(self.graduated and not was_graduated),
            "next_frontier_index": int(self.right_index),
        }


def install_ball_spawn_hook(raw: Any, curriculum: SpawnCurriculum) -> None:
    """Patch ``reset`` so the curriculum x is applied inside ``add_ball``."""
    if getattr(raw, "_spawn_curriculum_installed", False):
        raw._spawn_curriculum = curriculum
        return
    original_reset = raw.reset

    def reset_with_spawn(seed=None, options=None):
        original_add_ball = raw.add_ball

        def add_ball(x, y, b_mass, b_friction, b_elasticity):
            chosen = raw._spawn_curriculum.choose_x(raw, float(x))
            return original_add_ball(chosen, y, b_mass, b_friction, b_elasticity)

        raw.add_ball = add_ball
        try:
            return original_reset(seed=seed, options=options)
        finally:
            raw.add_ball = original_add_ball

    raw.reset = reset_with_spawn
    raw._spawn_curriculum = curriculum
    raw._spawn_curriculum_installed = True


class CurriculumTrainingEnvironment(TrainingEnvironment):
    """``TrainingEnvironment`` that spawns the ball through ``SpawnCurriculum``."""

    def __init__(self, args, *, rng=None):
        super().__init__(args, rng=rng)
        self.curriculum = SpawnCurriculum.from_args(args)
        install_ball_spawn_hook(self._raw, self.curriculum)

    def start_episode(self, seed: int):
        # Physics is applied by the parent before reset; the hook then moves x.
        return super().start_episode(seed)


def prepare_curriculum_episode(training_env: Any, global_step: int) -> None:
    curriculum = getattr(training_env, "curriculum", None)
    if curriculum is not None:
        curriculum.set_step(global_step)


def counts_toward_full_task(info: dict | None) -> bool:
    """Sample-efficiency threshold tracks the native task, not the short stages."""
    if info is None:
        return True
    return bool(info["graduated"]) and not bool(info["just_graduated"])


def on_curriculum_episode_end(
    training_env: Any,
    writer: Any,
    episode_context: dict | None,
    *,
    success: bool,
    global_step: int,
    learning_starts: int,
) -> tuple[dict, bool]:
    """Log the stage that just finished. Returns ``(context, count_toward_threshold)``."""
    logged = dict(episode_context or {})
    curriculum = getattr(training_env, "curriculum", None)
    if curriculum is None:
        return logged, True
    info = curriculum.note_outcome(
        success=success,
        global_step=global_step,
        learning_started=global_step >= learning_starts,
    )
    logged.update(info)
    if writer is not None:
        writer.add_scalar("curriculum/frontier_index", info["frontier_index"], global_step)
        if info["spawn_x"] is not None:
            writer.add_scalar("curriculum/spawn_x", info["spawn_x"], global_step)
        writer.add_scalar("curriculum/stage_success", info["stage_success"], global_step)
        writer.add_scalar("curriculum/graduated", float(info["graduated"]), global_step)
        writer.add_scalar("curriculum/native_spawn", float(info["native_spawn"]), global_step)
    if info["advanced"]:
        how = "forced" if info["forced"] else "mastered"
        if info["just_graduated"]:
            print(f"[curriculum @ {global_step}] graduated to native far spawn ({how})")
        else:
            print(
                f"[curriculum @ {global_step}] {how}: frontier piston {info['next_frontier_index']} "
                f"(stage success {info['stage_success']:.2f})"
            )
    return logged, counts_toward_full_task(info)


def curriculum_log_suffix(context: dict) -> str:
    if "spawn_x" not in context or context.get("spawn_x") is None:
        return ""
    return (
        f", spawn_x={float(context['spawn_x']):.0f}"
        f", frontier={int(context['frontier_index'])}"
        f", graduated={int(bool(context.get('graduated')))}"
    )


# Re-export so training scripts don't reach into the unwrapper themselves.
__all__ = [
    "CurriculumTrainingEnvironment",
    "SpawnCurriculum",
    "counts_toward_full_task",
    "curriculum_log_suffix",
    "install_ball_spawn_hook",
    "on_curriculum_episode_end",
    "prepare_curriculum_episode",
]
