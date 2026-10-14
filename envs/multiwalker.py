"""PettingZoo Multiwalker with in-place physics context (no per-episode recreate).

Default Box2D gravity is ``(0, -10)``, package fixture density is ``1.0``, and
terrain friction is ``2.5``. Those three scalars are the contextual variables:
training samples them inside an ID factor window; 1D eval slices include true OOD.

``reset`` rebuilds the Box2D world, so gravity / mass / friction must be applied
*after* every reset (same pattern as Pistonball gravity).
"""

from __future__ import annotations

import random
from typing import Any

from evaluation.context_sampling import (
    DEFAULT_EVAL_FACTOR_GRID,
    DEFAULT_ID_FACTOR_HIGH,
    DEFAULT_ID_FACTOR_LOW,
    ContextSpec,
    make_context_rng,
)

MULTIWALKER_PHYSICS_KEYS: tuple[str, ...] = (
    "gravity",
    "package_mass",
    "terrain_friction",
)
DEFAULT_GRAVITY = -10.0
DEFAULT_PACKAGE_MASS = 1.0
DEFAULT_TERRAIN_FRICTION = 2.5


def default_context(args) -> dict:
    return {
        "gravity": float(getattr(args, "gravity", DEFAULT_GRAVITY)),
        "package_mass": float(getattr(args, "package_mass", DEFAULT_PACKAGE_MASS)),
        "terrain_friction": float(getattr(args, "terrain_friction", DEFAULT_TERRAIN_FRICTION)),
    }


def context_spec_from_mw_args(args) -> ContextSpec:
    vary = tuple(getattr(args, "train_vary_params", None) or MULTIWALKER_PHYSICS_KEYS)
    eval_params = tuple(getattr(args, "eval_vary_params", None) or MULTIWALKER_PHYSICS_KEYS)
    grid = tuple(float(f) for f in getattr(args, "eval_factor_grid", DEFAULT_EVAL_FACTOR_GRID))
    if not grid:
        grid = DEFAULT_EVAL_FACTOR_GRID
    return ContextSpec(
        default=default_context(args),
        id_factor_low=float(getattr(args, "id_factor_low", DEFAULT_ID_FACTOR_LOW)),
        id_factor_high=float(getattr(args, "id_factor_high", DEFAULT_ID_FACTOR_HIGH)),
        vary_params=vary,
        eval_params=eval_params,
        eval_factor_grid=grid,
    )


def get_raw_multiwalker_env(env: Any) -> Any:
    """Unwrap SuperSuit / PettingZoo wrappers to ``MultiWalkerEnv`` (has ``walkers`` + ``world``)."""
    cur = env
    visited: set[int] = set()
    while id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "walkers") and hasattr(cur, "world"):
            return cur
        nxt = None
        for attr in ("unwrapped", "env", "aec_env", "par_env", "environment"):
            if hasattr(cur, attr):
                cand = getattr(cur, attr)
                if cand is not None and cand is not cur and id(cand) not in visited:
                    nxt = cand
                    break
        if nxt is None:
            break
        cur = nxt
    raise RuntimeError("Could not unwrap to MultiWalkerEnv (needed to update physics).")


class TimePenaltyWrapper:
    """Add a constant per-step reward (typically negative) to punish standing still.

    PettingZoo Multiwalker has no built-in time penalty; progress reward only
    moves when the package advances in x. A small negative ``time_penalty``
    (e.g. ``-0.1``) makes surviving 500 steps without walking costly (~-50).
    """

    def __init__(self, env: Any, time_penalty: float):
        self.env = env
        self.time_penalty = float(time_penalty)

    def step(self, action_dict: dict):
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        if self.time_penalty != 0.0 and rewards:
            rewards = {a: float(r) + self.time_penalty for a, r in rewards.items()}
        return obs, rewards, terms, truncs, infos

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_env(args, context: dict | None = None, render_mode: str | None = None):
    from pettingzoo.sisl import multiwalker_v9

    ctx = dict(context or default_context(args))
    env = multiwalker_v9.parallel_env(
        n_walkers=int(getattr(args, "n_walkers", 3)),
        position_noise=float(getattr(args, "position_noise", 1e-3)),
        angle_noise=float(getattr(args, "angle_noise", 1e-3)),
        forward_reward=float(getattr(args, "forward_reward", 1.0)),
        terminate_reward=float(getattr(args, "terminate_reward", -100.0)),
        fall_reward=float(getattr(args, "fall_reward", -10.0)),
        shared_reward=bool(getattr(args, "shared_reward", True)),
        terminate_on_fall=bool(getattr(args, "terminate_on_fall", True)),
        remove_on_fall=bool(getattr(args, "remove_on_fall", True)),
        terrain_length=int(getattr(args, "terrain_length", 75)),
        max_cycles=int(getattr(args, "max_cycles", 500)),
        render_mode=render_mode,
    )
    apply_context_to_env(env, ctx)
    penalty = float(getattr(args, "time_penalty", 0.0) or 0.0)
    if penalty != 0.0:
        env = TimePenaltyWrapper(env, penalty)
    return env


def apply_context_to_env(env: Any, context: dict) -> None:
    """Stash physics on the raw env; takes effect in ``apply_context_after_reset``."""
    raw = get_raw_multiwalker_env(env)
    raw._cmarl_gravity = float(context["gravity"])
    raw._cmarl_package_mass = float(context["package_mass"])
    raw._cmarl_terrain_friction = float(context["terrain_friction"])


def apply_context_after_reset(env: Any, context: dict | None = None) -> dict[str, float]:
    """Patch gravity / package mass / terrain friction on the live Box2D world."""
    raw = get_raw_multiwalker_env(env)
    if context is not None:
        apply_context_to_env(env, context)
    gy = float(getattr(raw, "_cmarl_gravity", DEFAULT_GRAVITY))
    dens = float(getattr(raw, "_cmarl_package_mass", DEFAULT_PACKAGE_MASS))
    fric = float(getattr(raw, "_cmarl_terrain_friction", DEFAULT_TERRAIN_FRICTION))
    raw.world.gravity = (0.0, gy)
    if raw.package is not None:
        for fixture in raw.package.fixtures:
            fixture.density = dens
        raw.package.ResetMassData()
    for body in raw.terrain or []:
        for fixture in body.fixtures:
            fixture.friction = fric
    mass = float(raw.package.mass) if raw.package is not None else float("nan")
    live_fric = float(raw.terrain[0].fixtures[0].friction) if raw.terrain and raw.terrain[0].fixtures else float("nan")
    return {
        "gravity": float(raw.world.gravity[1]),
        "package_mass": mass,
        "terrain_friction": live_fric,
    }


def stack_obs(obs_dict: dict, agents: list[str]) -> Any:
    import numpy as np

    return np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in agents], axis=0)


class TrainingEnvironment:
    """One long-lived Multiwalker env; resample physics in-place each episode."""

    def __init__(self, args, *, rng: random.Random | None = None):
        self.args = args
        self.regime = getattr(args, "training_regime", "static")
        self.spec = context_spec_from_mw_args(args)
        self.rng = rng if rng is not None else make_context_rng(int(args.seed))
        self.reference_context = dict(self.spec.default)
        self.current_context = dict(self.reference_context)
        self.env = make_env(args, self.current_context)
        self.episode_index = 0
        self._env_id = id(self.env)

    @property
    def env_recreated(self) -> bool:
        return id(self.env) != self._env_id

    def start_episode(self, seed: int) -> tuple[dict, dict]:
        if self.regime == "varying":
            mode = getattr(self.args, "train_context_sampling", "random")
            if mode == "grid":
                self.current_context = self.spec.sample_grid(self.rng)
            else:
                self.current_context = self.spec.sample_random(self.rng)
        else:
            self.current_context = dict(self.reference_context)
        apply_context_to_env(self.env, self.current_context)
        self.episode_index += 1
        obs_dict, _ = self.env.reset(seed=int(seed))
        apply_context_after_reset(self.env, self.current_context)
        return obs_dict, dict(self.current_context)

    def reset(self, seed: int) -> tuple[dict, dict]:
        return self.start_episode(seed)

    def step(self, action_dict: dict):
        return self.env.step(action_dict)

    @property
    def agents(self):
        return self.env.agents

    def close(self) -> None:
        self.env.close()
