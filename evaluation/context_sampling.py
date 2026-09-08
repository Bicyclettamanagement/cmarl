"""Training-time context sampling for contextual Pistonball experiments.

Supports comparing:
* **static** — fixed nominal physics (classic baseline)
* **varying** — physics randomized or grid-sampled each episode (hidden or explicit context)

Varying regimes **mutate physics on a single long-lived env** before ``reset``.
Recreating PettingZoo/pymunk envs every episode leaks host RAM and caused OOM on
32G nodes; do not call ``make_env`` per episode.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matd3_pistonball import Args, Context

TrainingRegime = Literal["static", "varying"]
TrainContextSampling = Literal["random", "grid"]

PHYSICS_CONTEXT_KEYS: tuple[str, ...] = (
    "ball_mass",
    "ball_friction",
    "ball_elasticity",
)


def training_regime_label(args) -> str:
    """Short tag used in run names (``static`` or ``varying``)."""
    return str(getattr(args, "training_regime", "static"))


def resolve_train_context_factors(args) -> tuple[float, ...]:
    """Factor range/endpoints used when sampling training contexts."""
    factors = getattr(args, "train_context_factors", None)
    if factors:
        return tuple(float(f) for f in factors)
    return tuple(float(f) for f in args.transfer_factors)


def sample_training_context(args, rng: random.Random | np.random.Generator) -> Context:
    """Sample a physics context for one training episode."""
    from matd3_pistonball import default_context

    if getattr(args, "training_regime", "static") != "varying":
        return default_context(args)

    base = default_context(args)
    mode: TrainContextSampling = getattr(args, "train_context_sampling", "random")

    if mode == "grid":
        from matd3_pistonball import build_transfer_contexts

        contexts = list(build_transfer_contexts(args).values())
        if isinstance(rng, np.random.Generator):
            idx = int(rng.integers(0, len(contexts)))
        else:
            idx = rng.randrange(len(contexts))
        return dict(contexts[idx])

    if mode == "random":
        factors = resolve_train_context_factors(args)
        lo, hi = min(factors), max(factors)
        ctx = dict(base)
        for key in PHYSICS_CONTEXT_KEYS:
            if isinstance(rng, np.random.Generator):
                factor = float(rng.uniform(lo, hi))
            else:
                factor = rng.uniform(lo, hi)
            ctx[key] = float(base[key]) * factor
        return ctx

    raise ValueError(f"unsupported train_context_sampling={mode!r}")


def get_raw_pistonball_env(env: Any) -> Any:
    """Unwrap SuperSuit / PettingZoo wrappers down to ``pistonball.raw_env``.

    Note: ``raw_env.space`` is only created after the first ``reset``, so we detect
    the raw env via ``ball_mass`` / ``add_ball`` rather than ``space``.
    """
    cur = env
    visited: set[int] = set()
    while id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "ball_mass") and hasattr(cur, "add_ball"):
            return cur
        nxt = None
        # Prefer ``unwrapped`` when present — it usually skips straight to raw_env.
        for attr in ("unwrapped", "env", "aec_env", "par_env", "environment"):
            if hasattr(cur, attr):
                cand = getattr(cur, attr)
                if cand is not None and cand is not cur and id(cand) not in visited:
                    nxt = cand
                    break
        if nxt is None:
            break
        cur = nxt
    raise RuntimeError(
        "Could not unwrap PettingZoo/SuperSuit stack to pistonball.raw_env "
        "(needed to update physics without recreating the environment)."
    )


def apply_context_to_env(env: Any, context: Context) -> None:
    """Write physics parameters onto the live raw env (takes effect on next ``reset``)."""
    raw = get_raw_pistonball_env(env)
    raw.time_penalty = float(context["time_penalty"])
    raw.ball_mass = float(context["ball_mass"])
    raw.ball_friction = float(context["ball_friction"])
    raw.ball_elasticity = float(context["ball_elasticity"])
    # Spawn flags are booleans read during reset; keep them consistent with context.
    if "random_drop" in context:
        raw.random_drop = bool(context["random_drop"])
    if "random_rotate" in context:
        raw.random_rotate = bool(context["random_rotate"])


class TrainingEnvironment:
    """Pistonball env wrapper that optionally resamples physics each episode.

    Creates **one** underlying env for the whole run. Under ``training_regime=varying``,
    each episode samples a new context, applies it in-place, then ``reset``s.
    """

    def __init__(self, args, *, rng: random.Random | np.random.Generator | None = None):
        from matd3_pistonball import default_context, make_env

        self.args = args
        self.regime: TrainingRegime = getattr(args, "training_regime", "static")
        self.rng = rng if rng is not None else random.Random(args.seed)
        self.reference_context: Context = default_context(args)
        self.current_context: Context = dict(self.reference_context)
        self.env = make_env(args, self.current_context)
        self._raw = get_raw_pistonball_env(self.env)
        self.episode_index = 0
        # Identity of the outer env object — used by tests to detect accidental recreation.
        self._env_id = id(self.env)

    @property
    def env_recreated(self) -> bool:
        """True if the outer env object was replaced (should stay False after init)."""
        return id(self.env) != self._env_id

    def start_episode(self, seed: int) -> tuple[dict, Context]:
        """Begin a new episode; update physics in-place when ``training_regime=varying``."""
        if self.regime == "varying":
            self.current_context = sample_training_context(self.args, self.rng)
            apply_context_to_env(self.env, self.current_context)
        else:
            self.current_context = dict(self.reference_context)
        self.episode_index += 1
        obs_dict, _ = self.env.reset(seed=seed)
        return obs_dict, dict(self.current_context)

    def reset(self, seed: int) -> tuple[dict, Context]:
        """Alias for ``start_episode`` (first episode of training)."""
        return self.start_episode(seed)

    def step(self, action_dict: dict):
        return self.env.step(action_dict)

    @property
    def agents(self):
        return self.env.agents

    def close(self) -> None:
        self.env.close()
