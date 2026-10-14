"""Watch trained policies in Pistonball / Multiwalker (RGB rollouts for notebooks).

The rliable notebook reports aggregate scores. This module records the actual
episode so you can see coordination, failures, and how physics changes behavior.

Two comparison modes:
* one experiment (all seeds) on several env configs — grid columns are physics
* several experiments (all seeds) on one env config — grid columns are seeds
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from evaluation.checkpoint_eval import (
    EVALUATION_RESULTS_FILENAME,
    detect_family,
    instantiate_args,
    load_algorithm,
)
from evaluation.context_sampling import (
    ALLOWED_PHYSICS_KEYS,
    EvalPoint,
    context_spec_from_args,
    eval_points_1d,
)
from evaluation.rliable_utils import (
    EXPERIMENT_PRESETS,
    RUN_MANIFEST_FILENAME,
    discover_run_dirs_relaxed,
    parse_seed_from_run_name,
)

# Headless pygame (cluster login node / Jupyter without a display).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

DEFAULT_GIF_WIDTH = 176
DEFAULT_FPS = 12
_CONTEXT_SHORT = {
    "ball_mass": "mass",
    "ball_friction": "fric",
    "ball_elasticity": "elas",
    "package_mass": "pkg",
    "terrain_friction": "terr",
    "gravity": "grav",
}
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


@dataclass
class LoadedPolicy:
    """Frozen actors plus the metadata needed to roll them out."""

    run_dir: Path
    label: str
    short_label: str
    family: str
    args: Any
    algo: Any
    agents: list[str]
    actors: torch.nn.ModuleList
    device: torch.device
    reference_context: dict
    seed: int
    checkpoint: str
    exp_name: str
    method_tag: str
    eval_return_by_context: dict[str, float] = field(default_factory=dict)
    eval_iqm_by_context: dict[str, float] = field(default_factory=dict)
    train_return: float | None = None

    @property
    def env_kind(self) -> str:
        return "multiwalker" if self.family == "multiwalker" else "pistonball"


@dataclass
class RolloutClip:
    """One recorded episode."""

    frames: np.ndarray  # (T, H, W, 3) uint8
    episode_return: float
    length: int
    success: float
    context_name: str
    context: dict
    policy_label: str
    short_label: str = ""
    seed: int = 0
    env_kind: str = "pistonball"
    physics_probe: dict[str, float] | None = None
    split: str = ""
    eval_return: float | None = None
    eval_iqm: float | None = None
    returns_at_frame: np.ndarray | None = None

    @property
    def title(self) -> str:
        label = self.short_label or self.policy_label
        bits = [f"{label} s{self.seed}", short_context_name(self.context_name)]
        if self.split:
            bits.append(self.split)
        return " · ".join(bits)

    @property
    def caption(self) -> str:
        kpi = _kpi_text(self.eval_iqm, self.eval_return)
        extra = f"  ·  {kpi}" if kpi else ""
        return f"R {self.episode_return:.1f}  ·  {self.length} steps{extra}"


def list_available_runs(runs_root: str | Path) -> list[dict[str, Any]]:
    """Summarize run folders that have a checkpoint (for the notebook picker)."""
    from evaluation.run_paths import iter_run_dirs

    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for child in iter_run_dirs(runs_root):
        ckpt_dir = child / "checkpoints"
        has_best = (ckpt_dir / "best.pt").exists()
        has_latest = (ckpt_dir / "latest.pt").exists()
        if not has_best and not has_latest:
            continue
        with open(child / RUN_MANIFEST_FILENAME) as f:
            manifest = json.load(f)
        args_dict = dict(manifest.get("args") or {})
        seed = args_dict.get("seed")
        if seed is None:
            seed = parse_seed_from_run_name(child.name)
        experiment = str(args_dict.get("experiment") or args_dict.get("exp_name") or child.name)
        preset = _preset_for_run(args_dict, child.name)
        try:
            rel = str(child.relative_to(runs_root))
        except ValueError:
            rel = str(child)
        rows.append(
            {
                "preset": preset or "—",
                "label": (EXPERIMENT_PRESETS.get(preset) or {}).get("label") or experiment,
                "exp_name": experiment,
                "seed": int(seed) if seed is not None else None,
                "family": detect_family(args_dict),
                "method_tag": args_dict.get("method_tag") or "—",
                "checkpoint": "best" if has_best else "latest",
                "finished": bool(manifest.get("finished")),
                "run_dir": str(child),
                "group": str(Path(rel).parent) if Path(rel).parent != Path(".") else "—",
            }
        )
    return rows


def suggested_contexts(env_kind: str) -> list[str]:
    """A small, readable set of ID + OOD configs to watch first."""
    if env_kind == "multiwalker":
        return ["default", "gravity_x0.5", "gravity_x2", "package_mass_x2"]
    return ["default", "ball_mass_x0.5", "ball_mass_x2", "ball_elasticity_x0.5"]


def list_context_names(policy: LoadedPolicy) -> list[str]:
    return [p.name for p in eval_points_1d(policy.args)]


def resolve_run_dirs(
    runs_root: str | Path,
    spec: str | Path,
    *,
    seeds: int | Sequence[int] | None = None,
) -> list[Path]:
    """Resolve a preset / exp-name / run directory to one path per seed."""
    from evaluation.run_paths import iter_run_dirs

    runs_root = Path(runs_root)
    spec_path = Path(spec)
    if spec_path.is_dir() and (spec_path / RUN_MANIFEST_FILENAME).exists():
        return [spec_path]
    under_root = runs_root / spec
    if under_root.is_dir() and (under_root / RUN_MANIFEST_FILENAME).exists():
        return [under_root]

    # Nested layout: allow passing just the run folder basename.
    if "__" in spec_path.name:
        exact = [child for child in iter_run_dirs(runs_root) if child.name == spec_path.name]
        if exact:
            return exact

    key = str(spec)
    exp_name = EXPERIMENT_PRESETS[key]["exp_name"] if key in EXPERIMENT_PRESETS else key
    seed_list = _as_seed_list(seeds)
    found = discover_run_dirs_relaxed(runs_root, exp_name, seeds=seed_list, require_finished=False)
    if not found:
        available = _available_experiment_names(runs_root)
        similar = [name for name in available if exp_name in name or name in exp_name]
        hint = ""
        if similar:
            hint = f" Similar on disk: {', '.join(similar[:8])}."
        elif available:
            sample = ", ".join(available[:8])
            more = f" (+{len(available) - 8} more)" if len(available) > 8 else ""
            hint = f" Available under {runs_root}: {sample}{more}."
        archived = Path("old_runs")
        if archived.is_dir() and discover_run_dirs_relaxed(archived, exp_name, seeds=seed_list):
            hint += f" Found under {archived}/ — pass runs_root=old_runs or pick a live exp_name."
        raise FileNotFoundError(
            f"no checkpointed run for {spec!r} (exp_name={exp_name!r}, seeds={seeds}) "
            f"under {runs_root}.{hint}"
        )
    return found


def _available_experiment_names(runs_root: Path) -> list[str]:
    from evaluation.run_paths import iter_run_dirs

    names: set[str] = set()
    for child in iter_run_dirs(runs_root):
        manifest_path = child / RUN_MANIFEST_FILENAME
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            args_dict = dict(manifest.get("args") or {})
            exp = str(args_dict.get("experiment") or args_dict.get("exp_name") or "")
            if exp:
                names.add(exp)
                continue
        # Fall back to folder prefix before the first ``__``.
        names.add(child.name.split("__", 1)[0])
    return sorted(names)


def resolve_run_dir(
    runs_root: str | Path,
    spec: str | Path,
    *,
    seed: int | None = None,
) -> Path:
    """Resolve to a single run (first match if ``seed`` is omitted)."""
    found = resolve_run_dirs(runs_root, spec, seeds=seed)
    return found[0]


def load_policies(
    runs_root: str | Path,
    specs: Sequence[str | Path],
    *,
    seeds: int | Sequence[int] | None = None,
    seed: int | None = None,
    checkpoint: str = "best",
    device: torch.device | None = None,
) -> list[LoadedPolicy]:
    """Load policies for each spec. ``seeds=None`` loads every seed (default)."""
    if seeds is None and seed is not None:
        seeds = seed
    policies: list[LoadedPolicy] = []
    for spec in specs:
        for run_dir in resolve_run_dirs(runs_root, spec, seeds=seeds):
            policies.append(load_policy(run_dir, checkpoint=checkpoint, device=device))
    return policies


def load_policy(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    device: torch.device | None = None,
    label: str | None = None,
) -> LoadedPolicy:
    """Load actors from ``checkpoints/{best,latest}.pt``."""
    import matd3_pistonball as base

    run_dir = Path(run_dir)
    with open(run_dir / RUN_MANIFEST_FILENAME) as f:
        manifest = json.load(f)
    args_dict = dict(manifest.get("args") or {})
    family = detect_family(args_dict)
    algo = load_algorithm(args_dict)
    args = instantiate_args(algo.Args, args_dict)
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu"
        )

    ckpt_dir = run_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        fallback = ckpt_dir / "best.pt" if checkpoint != "best" else ckpt_dir / "latest.pt"
        if not fallback.exists():
            raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
        ckpt_path = fallback
        checkpoint = ckpt_path.stem

    nets = algo.build_agents(args, device)
    base.load_checkpoint_into_nets(ckpt_path, nets, device)
    nets["actors"].eval()

    spec = context_spec_from_args(args)
    seed = int(args_dict.get("seed", getattr(args, "seed", 0)))
    experiment = str(args_dict.get("experiment") or args_dict.get("exp_name") or run_dir.name)
    preset = _preset_for_run(args_dict, run_dir.name)
    auto_label = (EXPERIMENT_PRESETS.get(preset) or {}).get("label") or experiment
    short_label = preset or experiment
    eval_mean, eval_iqm = _eval_kpis_from_run(run_dir)
    train_return = _train_return_from_manifest(manifest)
    return LoadedPolicy(
        run_dir=run_dir,
        label=label or f"{auto_label} (seed {seed})",
        short_label=short_label,
        family=family,
        args=args,
        algo=algo,
        agents=list(nets["agents"]),
        actors=nets["actors"],
        device=device,
        reference_context=dict(spec.default),
        seed=seed,
        checkpoint=checkpoint,
        exp_name=experiment,
        method_tag=str(getattr(args, "method_tag", "") or args_dict.get("method_tag") or ""),
        eval_return_by_context=eval_mean,
        eval_iqm_by_context=eval_iqm,
        train_return=train_return,
    )


def resolve_context(policy: LoadedPolicy, spec: str | dict | EvalPoint) -> EvalPoint:
    """Map a name (``default``, ``ball_mass_x2``, …) or dict onto an eval point."""
    if isinstance(spec, EvalPoint):
        return spec
    points = {p.name: p for p in eval_points_1d(policy.args)}
    if isinstance(spec, dict):
        ctx = dict(policy.reference_context)
        ctx.update(spec)
        name = spec.get("name") or _context_label(ctx, policy.reference_context)
        return EvalPoint(
            name=str(name),
            context=ctx,
            split="custom",
            varied_param=None,
            factor=1.0,
            region="custom",
        )
    name = str(spec)
    if name in points:
        return points[name]
    parsed = _parse_eval_name(name)
    if parsed is not None:
        param, factor = parsed
        spec_obj = context_spec_from_args(policy.args)
        if param not in spec_obj.default:
            raise KeyError(
                f"context {name!r} uses {param!r}, which is not a physics key for "
                f"{policy.env_kind}. Try one of: {list_context_names(policy)}"
            )
        ctx = spec_obj.scale(param, factor)
        id_factor = spec_obj.is_id_factor(factor)
        return EvalPoint(
            name=name,
            context=ctx,
            split="id" if id_factor else "ood",
            varied_param=param,
            factor=float(factor),
            region="interpolation" if id_factor else "extrapolation",
        )
    raise KeyError(f"unknown context {name!r}. Available: {list_context_names(policy)}")


@torch.no_grad()
def rollout_episode(
    policy: LoadedPolicy,
    context: str | dict | EvalPoint,
    *,
    seed: int | None = None,
    env=None,
    max_frames: int | None = None,
) -> RolloutClip:
    """Deterministic greedy rollout; returns RGB frames from ``env.render()``."""
    point = resolve_context(policy, context)
    owns_env = env is None
    if owns_env:
        env = _make_env(policy, point.context, render_mode="rgb_array")
    apply_ctx, apply_after = _context_fns(policy)
    apply_ctx(env, point.context)

    ep_seed = int(policy.seed if seed is None else seed)
    frames: list[np.ndarray] = []
    returns_at_frame: list[float] = []
    ep_return = 0.0
    ep_len = 0
    success = 0.0
    physics_probe: dict[str, float] | None = None
    try:
        obs_dict, _ = env.reset(seed=ep_seed)
        physics_probe = apply_after(env, point.context)
        frame = _render_rgb(env)
        if frame is not None:
            frames.append(frame)
            returns_at_frame.append(0.0)
        while env.agents:
            if max_frames is not None and ep_len >= int(max_frames):
                break
            if any(a not in obs_dict for a in policy.agents):
                break
            actions = _act(policy, obs_dict, env, point.context)
            action_dict = {a: actions[i] for i, a in enumerate(policy.agents) if a in env.agents}
            obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
            rew_vals = list(rewards.values())
            if rew_vals:
                ep_return += float(np.mean(rew_vals))
            ep_len += 1
            if terms and any(terms.values()):
                success = 1.0
            frame = _render_rgb(env)
            if frame is not None:
                frames.append(frame)
                returns_at_frame.append(float(ep_return))
            if not env.agents:
                break
            if truncs and all(truncs.get(a, False) or terms.get(a, False) for a in list(truncs)):
                break
    finally:
        if owns_env:
            env.close()

    if not frames:
        raise RuntimeError(
            "env.render() returned no RGB frames. For a headless machine this usually "
            "means pygame failed to initialize (SDL_VIDEODRIVER=dummy)."
        )
    return RolloutClip(
        frames=np.stack(frames, axis=0),
        episode_return=float(ep_return),
        length=int(ep_len),
        success=float(success),
        context_name=point.name,
        context=dict(point.context),
        policy_label=policy.label,
        short_label=policy.short_label,
        seed=int(policy.seed),
        env_kind=policy.env_kind,
        physics_probe=physics_probe,
        split=str(point.split or ""),
        eval_return=policy.eval_return_by_context.get(point.name, policy.train_return),
        eval_iqm=policy.eval_iqm_by_context.get(point.name),
        returns_at_frame=np.asarray(returns_at_frame, dtype=np.float32),
    )


def watch_one_policy_many_envs(
    policy: LoadedPolicy,
    contexts: Sequence[str | dict | EvalPoint],
    *,
    seed: int | None = None,
    max_frames: int | None = None,
) -> list[RolloutClip]:
    """Same agents, different physics — reuse one env and mutate in-place."""
    if not contexts:
        raise ValueError("contexts must be non-empty")
    first = resolve_context(policy, contexts[0])
    env = _make_env(policy, first.context, render_mode="rgb_array")
    clips: list[RolloutClip] = []
    try:
        for spec in contexts:
            clips.append(
                rollout_episode(
                    policy,
                    spec,
                    seed=seed,
                    env=env,
                    max_frames=max_frames,
                )
            )
    finally:
        env.close()
    return clips


def watch_many_policies_one_env(
    policies: Sequence[LoadedPolicy],
    context: str | dict | EvalPoint,
    *,
    seed: int | None = None,
    max_frames: int | None = None,
) -> list[RolloutClip]:
    """Different agents, same named physics config (resolved per policy)."""
    if not policies:
        raise ValueError("policies must be non-empty")
    kinds = {p.env_kind for p in policies}
    if len(kinds) > 1:
        raise ValueError(f"cannot mix env families in one comparison: {sorted(kinds)}")
    return [
        rollout_episode(policy, context, seed=seed, max_frames=max_frames) for policy in policies
    ]


def watch_specs(
    runs_root: str | Path,
    specs: Sequence[str | Path],
    contexts: Sequence[str | dict | EvalPoint],
    *,
    seeds: int | Sequence[int] | None = None,
    checkpoint: str = "best",
    device: torch.device | None = None,
    episode_seed: int = 0,
    max_frames: int | None = None,
) -> list[RolloutClip]:
    """Load each matching seed in turn and roll it out on every requested context.

    Policies are not kept in memory together, so comparing 5 seeds × 2 methods
    stays RAM-safe.
    """
    if not specs:
        raise ValueError("specs must be non-empty")
    if not contexts:
        raise ValueError("contexts must be non-empty")
    clips: list[RolloutClip] = []
    for spec in specs:
        run_dirs = resolve_run_dirs(runs_root, spec, seeds=seeds)
        for run_dir in run_dirs:
            policy = load_policy(run_dir, checkpoint=checkpoint, device=device)
            try:
                if len(contexts) == 1:
                    clips.append(
                        rollout_episode(
                            policy, contexts[0], seed=episode_seed, max_frames=max_frames
                        )
                    )
                else:
                    clips.extend(
                        watch_one_policy_many_envs(
                            policy, contexts, seed=episode_seed, max_frames=max_frames
                        )
                    )
            finally:
                del policy
    return clips


def short_context_name(name: str) -> str:
    if name == "default":
        return "default"
    parsed = _parse_eval_name(name)
    if parsed is None:
        return name
    param, factor = parsed
    stem = _CONTEXT_SHORT.get(param, param)
    return f"{stem}×{factor:g}"


def crop_empty_top(
    frame: np.ndarray,
    *,
    white_threshold: float = 245.0,
    keep_min: float = 0.5,
) -> np.ndarray:
    """Drop near-white sky above Pistonball; always keep at least ``keep_min`` of height."""
    h = int(frame.shape[0])
    if h < 8:
        return frame
    gray = frame.mean(axis=2)
    row_mean = gray.mean(axis=1)
    content = np.flatnonzero(row_mean < float(white_threshold))
    if content.size == 0:
        top = h // 2
    else:
        top = int(content[0])
    max_top = int(h * (1.0 - float(keep_min)))
    top = min(max(top, 0), max_top)
    if top <= 0:
        return frame
    return np.ascontiguousarray(frame[top:])


def prepare_display_frames(
    clip: RolloutClip,
    *,
    max_width: int = DEFAULT_GIF_WIDTH,
    crop: bool = True,
    overlay: bool = True,
) -> np.ndarray:
    """Crop / resize / stamp HUD onto a clip for GIF encoding."""
    frames = clip.frames
    if crop and getattr(clip, "env_kind", "pistonball") != "multiwalker":
        frames = np.stack([crop_empty_top(frame) for frame in frames])
    frames = np.stack([_resize_frame(frame, max_width) for frame in frames])
    if overlay:
        returns = getattr(clip, "returns_at_frame", None)
        out = []
        for i, frame in enumerate(frames):
            live = None
            if returns is not None and i < len(returns):
                live = float(returns[i])
            elif i == len(frames) - 1:
                live = float(getattr(clip, "episode_return", 0.0))
            out.append(_overlay_hud(frame, *_hud_lines(clip, live_return=live)))
        frames = np.stack(out)
    return frames


def encode_gif_bytes(
    frames: np.ndarray,
    *,
    fps: int = DEFAULT_FPS,
    max_width: int = DEFAULT_GIF_WIDTH,
) -> bytes:
    """Encode ``(T,H,W,3)`` uint8 frames as a looping GIF."""
    from PIL import Image

    seq = [_to_pil(_resize_frame(frame, max_width)) for frame in frames]
    if not seq:
        raise ValueError("no frames to encode")
    buf = io.BytesIO()
    duration = max(int(round(1000 / max(fps, 1))), 40)
    seq[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=seq[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


def clips_to_html(
    clips: Sequence[RolloutClip],
    *,
    fps: int = DEFAULT_FPS,
    max_width: int = DEFAULT_GIF_WIDTH,
    crop: bool = True,
    overlay: bool = True,
    columns: int | None = None,
) -> str:
    """Compact GIF grid. Overlay carries experiment / context / eval KPI."""
    ordered = list(clips)
    cols = int(columns) if columns else _auto_columns(ordered)
    cells: list[str] = []
    for clip in ordered:
        frames = prepare_display_frames(clip, max_width=max_width, crop=crop, overlay=overlay)
        gif = encode_gif_bytes(frames, fps=fps, max_width=max_width)
        b64 = base64.b64encode(gif).decode("ascii")
        caption = "" if overlay else (
            "<figcaption style='font:11px/1.3 sans-serif;color:#444;margin-top:3px;'>"
            f"{_esc(clip.title)}<br>{_esc(clip.caption)}</figcaption>"
        )
        cells.append(
            "<figure style='margin:0;text-align:left;'>"
            f"<img src='data:image/gif;base64,{b64}' width='{int(max_width)}' "
            "style='display:block;height:auto;image-rendering:auto;'/>"
            f"{caption}</figure>"
        )
    return (
        f"<div style='display:grid;grid-template-columns:repeat({cols},max-content);"
        "gap:8px 10px;align-items:start;'>"
        + "".join(cells)
        + "</div>"
    )


def show_clips(
    clips: Sequence[RolloutClip],
    *,
    fps: int = DEFAULT_FPS,
    max_width: int = DEFAULT_GIF_WIDTH,
    crop: bool = True,
    overlay: bool = True,
    columns: int | None = None,
):
    """Display clips in a Jupyter cell. Returns the HTML object."""
    html = clips_to_html(
        clips, fps=fps, max_width=max_width, crop=crop, overlay=overlay, columns=columns
    )
    try:
        from IPython.display import HTML, display

        obj = HTML(html)
        display(obj)
        return obj
    except Exception:
        return html


def _as_seed_list(seeds: int | Sequence[int] | None) -> list[int] | None:
    if seeds is None:
        return None
    if isinstance(seeds, (int, np.integer)):
        return [int(seeds)]
    return [int(s) for s in seeds]


def _eval_kpis_from_run(run_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    path = run_dir / EVALUATION_RESULTS_FILENAME
    means: dict[str, float] = {}
    iqms: dict[str, float] = {}
    if not path.exists():
        return means, iqms
    with open(path) as f:
        payload = json.load(f)
    for name, stats in (payload.get("contexts") or {}).items():
        if not isinstance(stats, dict):
            continue
        if stats.get("return_mean") is not None:
            means[str(name)] = float(stats["return_mean"])
        if stats.get("return_iqm") is not None:
            iqms[str(name)] = float(stats["return_iqm"])
    return means, iqms


def _train_return_from_manifest(manifest: dict) -> float | None:
    stats = ((manifest.get("best_eval") or {}).get("stats") or {})
    value = stats.get("return_mean")
    return float(value) if value is not None else None


def _kpi_text(eval_iqm: float | None, eval_return: float | None) -> str:
    if eval_iqm is not None:
        return f"eval {eval_iqm:.0f}"
    if eval_return is not None:
        return f"eval {eval_return:.0f}"
    return ""


def _hud_lines(clip: RolloutClip, *, live_return: float | None) -> tuple[str, str]:
    label = getattr(clip, "short_label", None) or getattr(clip, "policy_label", "?")
    seed = getattr(clip, "seed", None)
    head = f"{label} s{seed}" if seed is not None else str(label)
    ctx = short_context_name(getattr(clip, "context_name", "?"))
    split = getattr(clip, "split", "") or ""
    if split and split not in ("custom",):
        ctx = f"{ctx} {split}"
    live = "R —" if live_return is None else f"R {_fmt_return(live_return)}"
    kpi = _kpi_text(getattr(clip, "eval_iqm", None), getattr(clip, "eval_return", None))
    line2 = f"{ctx}  {live}" + (f"  {kpi}" if kpi else "")
    return head, line2


def _fmt_return(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _overlay_hud(frame: np.ndarray, line1: str, line2: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    w, h = im.size
    font = _hud_font(max(10, w // 15))
    pad = 2
    bbox1 = draw.textbbox((0, 0), line1, font=font)
    bbox2 = draw.textbbox((0, 0), line2, font=font)
    text_h = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + pad * 3
    bar_h = min(max(text_h, 16), max(h // 3, 16))
    draw.rectangle([0, 0, w, bar_h], fill=(0, 0, 0))
    draw.text((pad, pad), line1, fill=(255, 255, 255), font=font)
    y2 = pad + (bbox1[3] - bbox1[1]) + 1
    draw.text((pad, y2), line2, fill=(255, 220, 80), font=font)
    return np.asarray(im)


def _hud_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, int(size))
            except OSError:
                continue
    return ImageFont.load_default()


def _auto_columns(clips: Sequence[RolloutClip]) -> int:
    if not clips:
        return 1
    n_ctx = len({getattr(c, "context_name", "") for c in clips})
    if n_ctx > 1:
        return n_ctx
    n_seed = len({(getattr(c, "short_label", getattr(c, "policy_label", "")), getattr(c, "seed", i)) for i, c in enumerate(clips)})
    if n_seed > 1:
        by_exp: dict[str, int] = {}
        for clip in clips:
            key = getattr(clip, "short_label", None) or getattr(clip, "policy_label", "")
            by_exp[key] = by_exp.get(key, 0) + 1
        return max(by_exp.values())
    return max(1, min(len(clips), 6))


def _preset_for_run(args_dict: dict, run_name: str) -> str | None:
    from matd3_pistonball import sanitize_experiment_tag

    experiment = str(args_dict.get("experiment") or "")
    exp_name = str(args_dict.get("exp_name") or "")
    for key, meta in EXPERIMENT_PRESETS.items():
        target = str(meta.get("exp_name") or "")
        if not target:
            continue
        if target in (experiment, exp_name):
            return key
        safe = sanitize_experiment_tag(target)
        if run_name.startswith(f"{target}__") or run_name.startswith(f"{safe}__"):
            return key
        if run_name.startswith(f"pistonball__{target}__") or run_name.startswith(f"pistonball__{safe}__"):
            return key
    return None


def _make_env(policy: LoadedPolicy, context: dict, render_mode: str | None):
    if policy.family == "multiwalker":
        from envs.multiwalker import make_env as make_eval_env
    else:
        import matd3_pistonball as base

        make_eval_env = base.make_env
    return make_eval_env(policy.args, context, render_mode=render_mode)


def _context_fns(policy: LoadedPolicy):
    if policy.family == "multiwalker":
        from envs.multiwalker import apply_context_after_reset, apply_context_to_env
    else:
        from evaluation.context_sampling import apply_context_after_reset, apply_context_to_env
    return apply_context_to_env, apply_context_after_reset


def _act(policy: LoadedPolicy, obs_dict: dict, env: Any, context: dict) -> np.ndarray:
    agents = policy.agents
    actors = policy.actors
    device = policy.device
    algo = policy.algo
    args = policy.args
    if policy.family == "multiwalker":
        obs = torch.as_tensor(algo.stack_obs(obs_dict, agents), device=device)
        ctx_dim = algo.context_feature_dim(args)
        ctx_t = None
        if ctx_dim > 0:
            vec = algo.context_to_vector(context, policy.reference_context)
            ctx_t = torch.as_tensor(vec, device=device).float().unsqueeze(0)
        actions = algo.joint_policy_actions(actors, obs.unsqueeze(0), context=ctx_t).squeeze(0)
        return actions.cpu().numpy()

    import matd3_pistonball as base

    obs = torch.as_tensor(base.stack_obs(obs_dict, agents), device=device).float() / 255.0
    if policy.family == "pistonball_context":
        phys_dim = algo.physics_context_dim(args)
        phys_t = None
        if phys_dim > 0:
            vec = algo.context_to_vector(context, policy.reference_context)
            phys_t = torch.as_tensor(vec, device=device).float().unsqueeze(0)
        agent_np = algo.maybe_agent_features(env, agents, args)
        agent_t = (
            torch.as_tensor(agent_np, device=device).float().unsqueeze(0)
            if agent_np is not None
            else None
        )
        actions = algo.joint_policy_actions(
            actors, obs.unsqueeze(0), context=phys_t, agent_context=agent_t
        ).squeeze(0)
        return actions.cpu().numpy()
    actions = algo.joint_policy_actions(actors, obs.unsqueeze(0)).squeeze(0)
    return actions.cpu().numpy()


def _render_rgb(env: Any) -> np.ndarray | None:
    frame = None
    if hasattr(env, "render"):
        try:
            frame = env.render()
        except Exception:
            frame = None
    if not _is_rgb(frame):
        raw = _try_unwrap_raw(env)
        if raw is not None and raw is not env and hasattr(raw, "render"):
            try:
                frame = raw.render()
            except Exception:
                frame = None
    if not _is_rgb(frame):
        return None
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _is_rgb(frame: Any) -> bool:
    if frame is None:
        return False
    try:
        arr = np.asarray(frame)
    except Exception:
        return False
    return arr.ndim >= 2 and arr.size > 0


def _try_unwrap_raw(env: Any) -> Any | None:
    try:
        from evaluation.context_sampling import get_raw_pistonball_env

        return get_raw_pistonball_env(env)
    except Exception:
        pass
    try:
        from envs.multiwalker import get_raw_multiwalker_env

        return get_raw_multiwalker_env(env)
    except Exception:
        return None


def _resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    new_w = int(max_width)
    new_h = max(1, int(round(h * max_width / w)))
    y_idx = np.linspace(0, h - 1, new_h).astype(int)
    x_idx = np.linspace(0, w - 1, new_w).astype(int)
    return np.ascontiguousarray(frame[y_idx][:, x_idx])


def _to_pil(frame: np.ndarray):
    from PIL import Image

    return Image.fromarray(frame)


def _parse_eval_name(name: str) -> tuple[str, float] | None:
    if "_x" not in name:
        return None
    param, _, factor_s = name.rpartition("_x")
    if param not in ALLOWED_PHYSICS_KEYS:
        return None
    try:
        return param, float(factor_s)
    except ValueError:
        return None


def _context_label(context: dict, reference: dict) -> str:
    diffs = []
    for key, value in context.items():
        if key not in reference:
            continue
        ref = reference[key]
        if isinstance(value, (int, float)) and isinstance(ref, (int, float)) and abs(ref) > 1e-12:
            if abs(float(value) - float(ref)) > 1e-9:
                diffs.append(f"{key}_x{float(value) / float(ref):g}")
        elif value != ref:
            diffs.append(str(key))
    return diffs[0] if len(diffs) == 1 else ("+".join(diffs) if diffs else "custom")


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
