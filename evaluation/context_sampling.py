"""Training-time context sampling and ID / OOD evaluation grids.

The previous setup sampled training physics in ``[min(transfer_factors),
max(transfer_factors)]`` and then evaluated at those same endpoints, so the
"OOD" sweep was still in-distribution. This module makes the split explicit:

* **ID window** — multiplicative factor range around the default physics used
  for *training* (varying regime) and for in-distribution evaluation.
* **1D eval grid** — one physics parameter is scaled at a time; the other
  parameters stay at their default. Factors inside the ID window are
  interpolation; factors below/above it are true extrapolation (OOD).

Varying regimes **mutate physics on a single long-lived env** before ``reset``.
Recreating PettingZoo/pymunk envs every episode leaks host RAM; do not call
``make_env`` per episode (training *or* the dense eval sweep).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

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
# Gravity is env-only (Pistonball hardcodes space.gravity=750 on reset). It is
# not part of the concat feature vector, so existing checkpoints stay valid.
GRAVITY_KEY = "gravity"
DEFAULT_GRAVITY = 750.0
MULTIWALKER_PHYSICS_KEYS: tuple[str, ...] = (
    "gravity",
    "package_mass",
    "terrain_friction",
)
ALLOWED_PHYSICS_KEYS: frozenset[str] = frozenset(
    PHYSICS_CONTEXT_KEYS + (GRAVITY_KEY,) + MULTIWALKER_PHYSICS_KEYS
)

DEFAULT_ID_FACTOR_LOW = 0.75
DEFAULT_ID_FACTOR_HIGH = 1.5
DEFAULT_EVAL_FACTOR_GRID: tuple[float, ...] = (
    0.5,
    0.6,
    0.75,
    0.875,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
)
# Wider extrapolation for a second eval pass on frozen checkpoints (no retrain).
HARD_EVAL_FACTOR_GRID: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)
EVAL_GRID_PRESETS: dict[str, tuple[float, ...]] = {
    "standard": DEFAULT_EVAL_FACTOR_GRID,
    "hard": HARD_EVAL_FACTOR_GRID,
}
# Hard re-eval also sweeps gravity (eval-only; not a concat feature).
HARD_EVAL_PARAMS: tuple[str, ...] = PHYSICS_CONTEXT_KEYS + (GRAVITY_KEY,)

# Dedicated RNG stream offsets so physics sampling, replay, and rollouts do not
# silently consume each other's draws.
CONTEXT_RNG_OFFSET = 17_389
REPLAY_RNG_OFFSET = 1_000_003
ROLLOUT_RNG_OFFSET = 2_000_003
EVAL_SEED_OFFSET = 10_000
SWEEP_SEED_OFFSET = 20_000


@dataclass(frozen=True)
class EvalPoint:
    """One evaluation environment on a 1D (or future 2D) context slice."""

    name: str
    context: dict
    split: str  # "id" | "ood"
    varied_param: str | None
    factor: float
    factor_b: float | None = None
    varied_param_b: str | None = None
    region: str = "interpolation"  # interpolation | extrapolation | inter_extrapolation

    def meta(self) -> dict:
        return {
            "split": self.split,
            "varied_param": self.varied_param,
            "factor": self.factor,
            "factor_b": self.factor_b,
            "varied_param_b": self.varied_param_b,
            "region": self.region,
        }


@dataclass
class ContextSpec:
    """Default physics + ID window + eval grid used by train and eval."""

    default: dict
    id_factor_low: float = DEFAULT_ID_FACTOR_LOW
    id_factor_high: float = DEFAULT_ID_FACTOR_HIGH
    vary_params: tuple[str, ...] = PHYSICS_CONTEXT_KEYS
    eval_params: tuple[str, ...] = PHYSICS_CONTEXT_KEYS
    eval_factor_grid: tuple[float, ...] = DEFAULT_EVAL_FACTOR_GRID

    def __post_init__(self) -> None:
        if self.id_factor_low <= 0 or self.id_factor_high <= 0:
            raise ValueError("ID factor bounds must be positive")
        if self.id_factor_low >= self.id_factor_high:
            raise ValueError(
                f"id_factor_low ({self.id_factor_low}) must be < id_factor_high ({self.id_factor_high})"
            )
        allowed = set(ALLOWED_PHYSICS_KEYS)
        unknown = [p for p in (*self.vary_params, *self.eval_params) if p not in allowed]
        if unknown:
            raise ValueError(
                f"unsupported physics params {unknown}; choose from {tuple(sorted(allowed))}"
            )
        if any(f <= 0 for f in self.eval_factor_grid):
            raise ValueError("eval_factor_grid values must be positive")

    def is_id_factor(self, factor: float, *, atol: float = 1e-9) -> bool:
        return (self.id_factor_low - atol) <= float(factor) <= (self.id_factor_high + atol)

    def scale(self, param: str, factor: float, *, base: dict | None = None) -> dict:
        ctx = dict(base if base is not None else self.default)
        ctx[param] = float(self.default[param]) * float(factor)
        return ctx

    def id_factors(self) -> tuple[float, ...]:
        return tuple(f for f in self.eval_factor_grid if self.is_id_factor(f))

    def sample_random(self, rng: random.Random | np.random.Generator) -> dict:
        """Independent uniform factors in the ID window for each varied param."""
        ctx = dict(self.default)
        for key in self.vary_params:
            factor = _uniform(rng, self.id_factor_low, self.id_factor_high)
            ctx[key] = float(self.default[key]) * factor
        return ctx

    def sample_grid(self, rng: random.Random | np.random.Generator) -> dict:
        """Quantized ID sampling: each varied param drawn from the ID eval grid."""
        id_grid = self.id_factors()
        if not id_grid:
            raise ValueError("eval_factor_grid has no factors inside the ID window")
        ctx = dict(self.default)
        for key in self.vary_params:
            factor = id_grid[_randrange(rng, len(id_grid))]
            ctx[key] = float(self.default[key]) * factor
        return ctx

    def eval_points_1d(self) -> list[EvalPoint]:
        """Default context + one-parameter slices (C.2-style curves)."""
        points = [
            EvalPoint(
                name="default",
                context=dict(self.default),
                split="id",
                varied_param=None,
                factor=1.0,
                region="interpolation",
            )
        ]
        seen: set[str] = {"default"}
        for param in self.eval_params:
            for factor in self.eval_factor_grid:
                if abs(float(factor) - 1.0) < 1e-12:
                    continue
                name = format_eval_name(param, factor)
                if name in seen:
                    continue
                seen.add(name)
                id_factor = self.is_id_factor(factor)
                points.append(
                    EvalPoint(
                        name=name,
                        context=self.scale(param, factor),
                        split="id" if id_factor else "ood",
                        varied_param=param,
                        factor=float(factor),
                        region="interpolation" if id_factor else "extrapolation",
                    )
                )
        return points

    def eval_points_2d(self, param_a: str, param_b: str) -> list[EvalPoint]:
        """Cartesian slice over two physics parameters (for future 2D plots)."""
        if param_a == param_b:
            raise ValueError("2D eval requires two distinct parameters")
        for param in (param_a, param_b):
            if param not in PHYSICS_CONTEXT_KEYS:
                raise ValueError(f"unsupported physics param {param!r}")
        points: list[EvalPoint] = []
        for fa in self.eval_factor_grid:
            for fb in self.eval_factor_grid:
                ctx = self.scale(param_a, fa)
                ctx = self.scale(param_b, fb, base=ctx)
                a_id = self.is_id_factor(fa)
                b_id = self.is_id_factor(fb)
                if a_id and b_id:
                    split, region = "id", "interpolation"
                elif a_id or b_id:
                    split, region = "ood", "inter_extrapolation"
                else:
                    split, region = "ood", "extrapolation"
                name = f"{param_a}_x{float(fa):g}__{param_b}_x{float(fb):g}"
                points.append(
                    EvalPoint(
                        name=name,
                        context=ctx,
                        split=split,
                        varied_param=param_a,
                        factor=float(fa),
                        varied_param_b=param_b,
                        factor_b=float(fb),
                        region=region,
                    )
                )
        return points

    def context_meta(self, points: Sequence[EvalPoint] | None = None) -> dict[str, dict]:
        pts = list(points) if points is not None else self.eval_points_1d()
        return {p.name: p.meta() for p in pts}


def format_eval_name(param: str, factor: float) -> str:
    return f"{param}_x{float(factor):g}"


def _uniform(rng: random.Random | np.random.Generator, lo: float, hi: float) -> float:
    if isinstance(rng, np.random.Generator):
        return float(rng.uniform(lo, hi))
    return float(rng.uniform(lo, hi))


def _randrange(rng: random.Random | np.random.Generator, n: int) -> int:
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(0, n))
    return int(rng.randrange(n))


def make_context_rng(seed: int) -> random.Random:
    """Deterministic physics-sampling stream, independent of env reset seeds."""
    return random.Random(int(seed) + CONTEXT_RNG_OFFSET)


def make_replay_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + REPLAY_RNG_OFFSET)


def make_rollout_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + ROLLOUT_RNG_OFFSET)


def context_spec_from_args(args) -> ContextSpec:
    """Build a ``ContextSpec`` from training / eval CLI args."""
    env_name = str(getattr(args, "env_name", "pistonball"))
    if env_name == "multiwalker":
        from envs.multiwalker import context_spec_from_mw_args

        return context_spec_from_mw_args(args)

    from matd3_pistonball import default_context

    vary = tuple(getattr(args, "train_vary_params", PHYSICS_CONTEXT_KEYS) or PHYSICS_CONTEXT_KEYS)
    eval_params = tuple(getattr(args, "eval_vary_params", None) or PHYSICS_CONTEXT_KEYS)
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


def training_regime_label(args) -> str:
    """Short tag used in run names (``static`` or ``varying``)."""
    return str(getattr(args, "training_regime", "static"))


def sample_training_context(args, rng: random.Random | np.random.Generator) -> dict:
    """Sample a physics context for one training episode (ID window only)."""
    spec = context_spec_from_args(args)
    if getattr(args, "training_regime", "static") != "varying":
        return dict(spec.default)

    mode: TrainContextSampling = getattr(args, "train_context_sampling", "random")
    if mode == "random":
        return spec.sample_random(rng)
    if mode == "grid":
        return spec.sample_grid(rng)
    raise ValueError(f"unsupported train_context_sampling={mode!r}")


def eval_points_1d(args) -> list[EvalPoint]:
    return context_spec_from_args(args).eval_points_1d()


def eval_contexts_1d(args) -> dict[str, dict]:
    """Name → context dict for the 1D ID/OOD eval grid (includes ``default``)."""
    return {p.name: p.context for p in eval_points_1d(args)}


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


def apply_context_to_env(env: Any, context: dict) -> None:
    """Write physics parameters onto the live raw env (takes effect on next ``reset``).

    Pistonball reads ``ball_mass`` / friction / elasticity in ``add_ball`` during
    ``reset``. Gravity is hardcoded inside ``reset``, so call
    ``apply_context_after_reset`` after every ``reset``.
    """
    raw = get_raw_pistonball_env(env)
    raw.time_penalty = float(context["time_penalty"])
    raw.ball_mass = float(context["ball_mass"])
    raw.ball_friction = float(context["ball_friction"])
    raw.ball_elasticity = float(context["ball_elasticity"])
    if GRAVITY_KEY in context:
        raw.gravity = float(context[GRAVITY_KEY])
    if "random_drop" in context:
        raw.random_drop = bool(context["random_drop"])
    if "random_rotate" in context:
        raw.random_rotate = bool(context["random_rotate"])


def apply_context_after_reset(env: Any, context: dict | None = None) -> dict[str, float]:
    """Patch gravity (reset overwrites it) and return the live pymunk probe."""
    raw = get_raw_pistonball_env(env)
    gy = DEFAULT_GRAVITY
    if context is not None and GRAVITY_KEY in context:
        gy = float(context[GRAVITY_KEY])
    elif hasattr(raw, "gravity"):
        gy = float(raw.gravity)
    raw.space.gravity = (0.0, gy)
    shapes = list(raw.ball.shapes)
    friction = float(shapes[0].friction) if shapes else float("nan")
    elasticity = float(shapes[0].elasticity) if shapes else float("nan")
    return {
        "ball_mass": float(raw.ball.mass),
        "ball_friction": friction,
        "ball_elasticity": elasticity,
        GRAVITY_KEY: float(raw.space.gravity[1]),
    }


def resolve_eval_grid(
    *,
    preset: str | None = None,
    factors: tuple[float, ...] | list[float] | None = None,
) -> tuple[float, ...]:
    if factors:
        return tuple(float(f) for f in factors)
    if preset:
        key = str(preset).strip().lower()
        if key not in EVAL_GRID_PRESETS:
            known = ", ".join(sorted(EVAL_GRID_PRESETS))
            raise ValueError(f"unknown eval grid preset {preset!r}; choose from: {known}")
        return EVAL_GRID_PRESETS[key]
    return DEFAULT_EVAL_FACTOR_GRID


def resolve_eval_vary_params(
    *,
    preset: str | None = None,
    params: tuple[str, ...] | Sequence[str] | None = None,
    env_name: str | None = None,
) -> tuple[str, ...] | None:
    """CLI override for 1D eval axes. ``hard`` adds gravity unless ``params`` is set."""
    if params:
        return tuple(str(p) for p in params)
    if preset and str(preset).strip().lower() == "hard":
        if str(env_name or "").lower() == "multiwalker":
            return MULTIWALKER_PHYSICS_KEYS
        return HARD_EVAL_PARAMS
    return None


class TrainingEnvironment:
    """Pistonball env wrapper that optionally resamples physics each episode.

    Creates **one** underlying env for the whole run. Under ``training_regime=varying``,
    each episode samples a new *in-distribution* context, applies it in-place, then
    ``reset``s.
    """

    def __init__(self, args, *, rng: random.Random | np.random.Generator | None = None):
        from matd3_pistonball import make_env

        self.args = args
        self.regime: TrainingRegime = getattr(args, "training_regime", "static")
        self.spec = context_spec_from_args(args)
        self.rng = rng if rng is not None else make_context_rng(int(args.seed))
        self.reference_context: dict = dict(self.spec.default)
        self.current_context: dict = dict(self.reference_context)
        self.env = make_env(args, self.current_context)
        self._raw = get_raw_pistonball_env(self.env)
        self.episode_index = 0
        self._env_id = id(self.env)

    @property
    def env_recreated(self) -> bool:
        """True if the outer env object was replaced (should stay False after init)."""
        return id(self.env) != self._env_id

    def start_episode(self, seed: int) -> tuple[dict, dict]:
        """Begin a new episode; update physics in-place when ``training_regime=varying``."""
        if self.regime == "varying":
            self.current_context = sample_training_context(self.args, self.rng)
            apply_context_to_env(self.env, self.current_context)
        else:
            self.current_context = dict(self.reference_context)
            apply_context_to_env(self.env, self.current_context)
        self.episode_index += 1
        obs_dict, _ = self.env.reset(seed=int(seed))
        apply_context_after_reset(self.env, self.current_context)
        return obs_dict, dict(self.current_context)

    def reset(self, seed: int) -> tuple[dict, dict]:
        """Alias for ``start_episode`` (first episode of training)."""
        return self.start_episode(seed)

    def step(self, action_dict: dict):
        return self.env.step(action_dict)

    @property
    def agents(self):
        return self.env.agents

    def close(self) -> None:
        self.env.close()


def summarize_context_sweep(per_context: dict[str, dict], points: Sequence[EvalPoint]) -> dict:
    """ID vs OOD gap from a 1D (or 2D) eval sweep keyed by ``EvalPoint.name``."""
    id_names = [p.name for p in points if p.split == "id"]
    ood_names = [p.name for p in points if p.split == "ood"]

    def _mean(names: list[str], key: str) -> float:
        vals = [per_context[n][key] for n in names if n in per_context and key in per_context[n]]
        return float(np.mean(vals)) if vals else float("nan")

    id_mean = _mean(id_names, "return_mean")
    ood_mean = _mean(ood_names, "return_mean")
    id_iqm = _mean(id_names, "return_iqm")
    ood_iqm = _mean(ood_names, "return_iqm")
    default_stats = per_context.get("default") or per_context.get("train") or {}
    return {
        "id_return_mean": id_mean,
        "ood_return_mean": ood_mean,
        "generalization_gap_mean": id_mean - ood_mean,
        "id_return_iqm": id_iqm,
        "ood_return_iqm": ood_iqm,
        "generalization_gap_iqm": id_iqm - ood_iqm,
        "train_return_mean": float(default_stats.get("return_mean", id_mean)),
        "train_return_iqm": float(default_stats.get("return_iqm", id_iqm)),
        "train_return": float(default_stats.get("return_mean", id_mean)),
        "ood_return_mean_legacy": ood_mean,
        "generalization_gap": id_mean - ood_mean,
    }
