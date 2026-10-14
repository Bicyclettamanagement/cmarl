"""Keyboard diagnostic for Pistonball: you drive some pistons, a checkpoint drives the rest.

The ball has to reach the left wall. ``piston_0`` is that wall; higher indices are
to the right, where the ball spawns when the drop is fixed. A positive action
raises a piston, a negative action lowers it, and zero holds its current height.
Raising the piston under the ball (or just to its right) slopes the ball left.
A piston left fully up is a wall; one left fully down is a pit.

CPU is enough. Each step is one forward pass of the shared actor (an 84×84
frame stack, feature width 128). Pygame does not use CUDA. Load one checkpoint
at a time and do not build the replay buffer.

Two modes share the same action mixing:

* interactive — a window on a machine with a display
* scripted — ``assist``, ``block``, ``pit``, ``freeze``, or ``policy``, so the
  same intervention can be repeated and so the cluster can test the loop
  without a display
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.context_sampling import get_raw_pistonball_env
from evaluation.rollout_viz import (
    LoadedPolicy,
    _act,
    _context_fns,
    _make_env,
    _render_rgb,
    resolve_context,
)

# Physics names accepted by ``resolve_context``, including factors off the
# training grid. ``[`` / ``]`` walk this list.
PHYSICS_CYCLE: tuple[str, ...] = (
    "default",
    "ball_mass_x0.25",
    "ball_mass_x0.5",
    "ball_mass_x2",
    "ball_mass_x4",
    "ball_friction_x0.25",
    "ball_friction_x2",
    "ball_friction_x4",
    "ball_elasticity_x0.25",
    "ball_elasticity_x2",
    "ball_elasticity_x4",
    "gravity_x0.25",
    "gravity_x2",
    "gravity_x4",
)

SCRIPTS: tuple[str, ...] = ("policy", "assist", "block", "pit", "freeze")

SUGGESTED_SESSIONS = """
Suggested sessions (n=5). piston_0 is the left wall, the goal.
Keep --episode-seed fixed when you compare two interventions.

Do not play all 20 seeds. Offline default-context return already splits them:

  strong, use for obstacles (return has little room to rise, so you watch
  whether teammates react):
    baseline5_varying_randdrop          seed 1    hidden, varying, eval ~94
    context5_concat_varying_randdrop    seed 1    concat, varying, eval ~92

  weak, use for "does a human help" (return has room):
    baseline5_varying_randdrop          seed 3    hidden, varying, eval ~41
    context5_concat_static_randdrop     seed 4    concat, static,  eval ~27

  skip the other n=5 seeds unless those four disagree, then repeat the same
  protocol on varying seed 2 of both methods. Touch n=8 only after that:
  one seed exists for baseline8_varying_randdrop and
  context8_concat_varying_randdrop.

1. Policy alone, then you cooperate.
   On the weak hidden run, first:
     --exp-name baseline5_varying_randdrop --seed 3 --script policy --episode-seed 0
   Then the same command with --script assist --human center, then the window
   with no --script and the center piston claimed. Raise a piston when the ball
   is on it or to its right; lower pistons that are already to the left of the
   ball. If you and the assist script both beat --script policy, the learned
   pistons are failing a local "rise under the ball" reflex. That is not a
   context-integration failure.

2. Block the ball.
   On a strong run (hidden varying seed 1, then concat varying seed 1):
     --script block --human 2 --episode-seed 0
   and the same thing by hand (hold W on piston_2 until it tops out, then let
   go; height is held by action 0). Read "near-obstacle policy mean" in the
   log. A large positive value means the neighbors still try to ramp. A value
   near 0 means they do not compensate. Compare hidden vs concat: the same
   reaction on both is evidence that the concat vector is not changing the
   coordination strategy.

3. Dig a pit (--script pit --human 2, or hold S).
   A hole asks for the opposite action of a wall. If neighbors never lower
   themselves to bridge it, the policy is a fixed "push up" wave. That wave
   does not use contact parameters, which matches the flat mass and elasticity
   curves.

4. Freeze one piston (--script freeze --human 2).
   Action 0 leaves the random initial height in place. If the ball stops there
   on the strong policy, teammates are not covering for a dead piston, so
   coordination is local. If they ramp over it, coordination exists and the
   flat physics curves are because that strategy ignores mass, friction, and
   elasticity.

5. Lie to a concat policy (key L, or --lie-context default).
   Only context5_* runs read a context vector. Set ball_mass_x4 in the window
   ([ and ]), then press L so the actor is fed the training default while the
   ball is actually 4× heavier. If the motion does not change, the actor is
   ignoring the concatenated features. Repeat at gravity_x4: gravity is not in
   the concat vector at all, so a change there would be the pixels, not the
   context input. Static concat seed 4 is the one that never trained on other
   physics; varying concat seed 1 did, and still posted flat curves.

6. Leave n=8 and the remaining seeds alone until 1–5 are written down.
"""


@dataclass
class ControlState:
    """Mutable keyboard / script state for one session."""

    manual: set[int]
    selected: int
    paused: bool = False
    done: bool = False
    quit: bool = False
    reset: bool = False
    new_layout: bool = False
    context_index: int = 0
    lie: bool = False
    held_action: float = 0.0


@dataclass
class EpisodeResult:
    """One play episode, scripted or interactive."""

    context_name: str
    fed_context_name: str
    seed: int
    manual: list[int]
    script: str | None
    episode_return: float
    length: int
    success: float
    truncated: bool
    physics: dict[str, float]
    random_drop: bool
    policy_mean: list[float]
    applied_mean: list[float]
    near_obstacle_policy_mean: float | None
    policy_actions: np.ndarray = field(repr=False)
    applied_actions: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context_name,
            "fed_context": self.fed_context_name,
            "seed": self.seed,
            "manual": self.manual,
            "script": self.script,
            "return": self.episode_return,
            "length": self.length,
            "success": self.success,
            "truncated": self.truncated,
            "physics": self.physics,
            "random_drop": self.random_drop,
            "policy_mean": self.policy_mean,
            "applied_mean": self.applied_mean,
            "near_obstacle_policy_mean": self.near_obstacle_policy_mean,
        }

    def format_line(self) -> str:
        manual = ",".join(str(i) for i in self.manual) or "—"
        reaction = (
            "n/a"
            if self.near_obstacle_policy_mean is None
            else f"{self.near_obstacle_policy_mean:+.2f}"
        )
        phys = self.physics
        return (
            f"{self.context_name}  fed={self.fed_context_name}  "
            f"R={self.episode_return:.1f}  steps={self.length}  "
            f"success={int(self.success)}  manual=[{manual}]  "
            f"near-obstacle policy mean={reaction}\n"
            f"  policy wanted: {_fmt_row(self.policy_mean)}\n"
            f"  applied:       {_fmt_row(self.applied_mean)}\n"
            f"  mass={phys.get('ball_mass', float('nan')):.3f}  "
            f"fric={phys.get('ball_friction', float('nan')):.3f}  "
            f"elas={phys.get('ball_elasticity', float('nan')):.3f}  "
            f"grav={phys.get('gravity', float('nan')):.0f}"
        )


def _fmt_row(values: list[float]) -> str:
    return " ".join(f"{v:+.2f}" for v in values)


def parse_piston_spec(spec: str, n: int) -> list[int]:
    """Parse ``center``, ``left``, ``right``, ``all``, ``2``, or ``1,3``.

    Indices are 0-based. ``piston_2`` is accepted. ``center`` is ``n // 2``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    text = spec.strip().lower()
    if text in ("", "none"):
        return []
    if text in ("center", "mid", "middle"):
        return [n // 2]
    if text == "left":
        return [0]
    if text == "right":
        return [n - 1]
    if text == "all":
        return list(range(n))
    out: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("piston_"):
            part = part.split("_", 1)[1]
        idx = int(part)
        if idx < 0 or idx >= n:
            raise ValueError(f"piston index {idx} is outside 0..{n - 1}")
        if idx not in out:
            out.append(idx)
    return out


def apply_drop_flags(context: dict, *, fixed_drop: bool) -> dict:
    """Copy ``context`` and optionally pin the ball to the right-hand spawn."""
    ctx = dict(context)
    if fixed_drop:
        ctx["random_drop"] = False
        ctx["random_rotate"] = False
    return ctx


def scripted_scalar(mode: str, raw: Any, index: int) -> float:
    """Action in ``[-1, 1]`` for one piston under a named script.

    ``assist`` raises when the ball is at or to the right of the piston (the
    direction that slopes the ball toward the left wall) and lowers otherwise.
    ``block`` holds fully up, ``pit`` fully down, ``freeze`` holds height.
    """
    if mode == "block":
        return 1.0
    if mode == "pit":
        return -1.0
    if mode == "freeze":
        return 0.0
    if mode == "assist":
        ball_x = float(raw.ball.position[0])
        piston_x = float(raw.pistonList[index].position[0])
        width = float(getattr(raw, "piston_width", 0.0))
        if ball_x >= piston_x - 0.5 * width:
            return 1.0
        return -1.0
    raise ValueError(f"unknown script {mode!r}; choose from {SCRIPTS}")


def apply_overrides(policy_actions: np.ndarray, overrides: dict[int, float]) -> np.ndarray:
    """Return a copy of ``policy_actions`` with some pistons replaced.

    ``policy_actions`` is ``(N,)`` or ``(N, act_dim)`` as produced by ``_act``.
    """
    original_shape = np.shape(policy_actions)
    if not original_shape:
        raise ValueError("policy_actions must contain one row per piston")
    flat = np.array(policy_actions, dtype=np.float32, copy=True).reshape(original_shape[0], -1)
    n = flat.shape[0]
    for index, value in overrides.items():
        if index < 0 or index >= n:
            raise IndexError(f"piston index {index} is outside 0..{n - 1}")
        flat[index, 0] = np.float32(value)
    return flat.reshape(original_shape)


def ball_piston_index(raw: Any) -> int:
    """Index of the piston whose center is closest to the ball."""
    ball_x = float(raw.ball.position[0])
    centers = [float(body.position[0]) for body in raw.pistonList]
    return int(np.argmin([abs(ball_x - x) for x in centers]))


def near_obstacle_policy_mean(
    policy_actions: np.ndarray,
    ball_index: np.ndarray,
    manual: set[int],
) -> float | None:
    """Mean policy output of non-manual pistons while the ball is next to a manual one.

    ``policy_actions`` is ``(T, N)`` of what the network wanted, including on
    pistons the human overrode. ``None`` when there is no manual piston or the
    ball never came within one piston of one.
    """
    if not manual or policy_actions.size == 0:
        return None
    actions = np.asarray(policy_actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    ball = np.asarray(ball_index, dtype=np.int32)
    near = np.zeros(len(ball), dtype=bool)
    for index in manual:
        near |= np.abs(ball - int(index)) <= 1
    others = [i for i in range(actions.shape[1]) if i not in manual]
    if not others or not near.any():
        return None
    return float(actions[near][:, others].mean())


def _action_dict(policy: LoadedPolicy, env: Any, actions: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(actions, dtype=np.float32).reshape(len(policy.agents), -1)
    return {
        agent: flat[i]
        for i, agent in enumerate(policy.agents)
        if agent in env.agents
    }


def _mean_first(actions: np.ndarray) -> list[float]:
    if actions.size == 0:
        return []
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        arr = arr[..., 0]
    return [float(x) for x in arr.mean(axis=0)]


def run_scripted_episode(
    policy: LoadedPolicy,
    env: Any,
    env_context: dict,
    fed_context: dict,
    *,
    context_name: str,
    fed_context_name: str,
    seed: int,
    manual: set[int],
    script: str,
    max_steps: int | None = None,
    collect_frames: bool = False,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    """Roll one episode. ``script='policy'`` leaves every piston to the network."""
    if policy.env_kind != "pistonball":
        raise ValueError(f"pistonball play only supports pistonball checkpoints, got {policy.family}")
    if script not in SCRIPTS:
        raise ValueError(f"unknown script {script!r}; choose from {SCRIPTS}")
    active_manual = set() if script == "policy" else set(manual)
    apply_ctx, apply_after = _context_fns(policy)
    apply_ctx(env, env_context)

    policy_rows: list[np.ndarray] = []
    applied_rows: list[np.ndarray] = []
    ball_rows: list[int] = []
    frames: list[np.ndarray] = []
    ep_return = 0.0
    success = 0.0
    length = 0
    truncated = False

    obs_dict, _ = env.reset(seed=int(seed))
    physics = apply_after(env, env_context)
    raw = get_raw_pistonball_env(env)
    random_drop = bool(getattr(raw, "random_drop", True))
    if collect_frames:
        frame = _render_rgb(env)
        if frame is not None:
            frames.append(frame)

    while env.agents:
        if max_steps is not None and length >= int(max_steps):
            truncated = True
            break
        if any(agent not in obs_dict for agent in policy.agents):
            truncated = True
            break
        with torch.no_grad():
            policy_act = _act(policy, obs_dict, env, fed_context)
        overrides: dict[int, float] = {}
        if script != "policy":
            overrides = {index: scripted_scalar(script, raw, index) for index in active_manual}
        applied = apply_overrides(policy_act, overrides)
        policy_rows.append(np.asarray(policy_act, dtype=np.float32).reshape(len(policy.agents), -1)[:, 0])
        applied_rows.append(np.asarray(applied, dtype=np.float32).reshape(len(policy.agents), -1)[:, 0])
        ball_rows.append(ball_piston_index(raw))

        obs_dict, rewards, terms, truncs, _ = env.step(_action_dict(policy, env, applied))
        if rewards:
            ep_return += float(np.mean(list(rewards.values())))
        length += 1
        if terms and any(terms.values()):
            success = 1.0
        if collect_frames:
            frame = _render_rgb(env)
            if frame is not None:
                frames.append(frame)
        if not env.agents:
            break
        if truncs and all(truncs.get(agent, False) or terms.get(agent, False) for agent in list(truncs)):
            break

    policy_arr = np.stack(policy_rows, axis=0) if policy_rows else np.zeros((0, len(policy.agents)))
    applied_arr = np.stack(applied_rows, axis=0) if applied_rows else np.zeros((0, len(policy.agents)))
    ball_arr = np.asarray(ball_rows, dtype=np.int32)
    result = EpisodeResult(
        context_name=context_name,
        fed_context_name=fed_context_name,
        seed=int(seed),
        manual=sorted(active_manual),
        script=script,
        episode_return=float(ep_return),
        length=int(length),
        success=float(success),
        truncated=bool(truncated),
        physics={k: float(v) for k, v in physics.items()},
        random_drop=random_drop,
        policy_mean=_mean_first(policy_arr),
        applied_mean=_mean_first(applied_arr),
        near_obstacle_policy_mean=near_obstacle_policy_mean(policy_arr, ball_arr, active_manual),
        policy_actions=policy_arr,
        applied_actions=applied_arr,
    )
    return result, frames


def append_log(path: Path, result: EpisodeResult, *, policy: LoadedPolicy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = result.as_dict()
    row["run_dir"] = str(policy.run_dir)
    row["label"] = policy.label
    row["checkpoint"] = policy.checkpoint
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def save_gif(path: Path, frames: list[np.ndarray], *, fps: int = 12) -> None:
    from evaluation.rollout_viz import encode_gif_bytes

    if not frames:
        raise RuntimeError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_gif_bytes(np.stack(frames, axis=0), fps=fps, max_width=480))


def prepare_pair(
    policy: LoadedPolicy,
    context_name: str,
    *,
    fixed_drop: bool,
    lie_context: str | None,
) -> tuple[dict, dict, str, str]:
    """Env context and the context vector the actor actually receives."""
    env_point = resolve_context(policy, context_name)
    env_ctx = apply_drop_flags(env_point.context, fixed_drop=fixed_drop)
    if lie_context:
        fed_point = resolve_context(policy, lie_context)
        fed_ctx = apply_drop_flags(fed_point.context, fixed_drop=fixed_drop)
        fed_name = fed_point.name
    else:
        fed_ctx = dict(env_ctx)
        fed_name = env_point.name
    return env_ctx, fed_ctx, env_point.name, fed_name


def make_play_env(policy: LoadedPolicy, context: dict) -> Any:
    """Offscreen env. The interactive window blits ``rgb_array`` frames itself.

    Pistonball's own ``human`` mode calls ``clock.tick`` once per piston, which
    makes a joint step several times slower than the physics rate. One blit per
    joint step keeps keyboard control responsive on a CPU.
    """
    return _make_env(policy, context, render_mode="rgb_array")


def _panel_lines(state: ControlState, result_bits: dict[str, Any], policy: LoadedPolicy) -> list[str]:
    manual = sorted(state.manual)
    claimed = ",".join(f"piston_{i}" for i in manual) or "none"
    lie = "ON (actor sees default physics)" if state.lie else "off"
    status = "PAUSED" if state.paused else ("EPISODE OVER — R resets" if state.done else "running")
    lines = [
        status,
        f"{policy.short_label}  seed {policy.seed}  ckpt {policy.checkpoint}",
        f"env {result_bits.get('context', '')}   actor fed {result_bits.get('fed', '')}",
        f"lie {lie}    drop {'fixed' if result_bits.get('fixed_drop') else 'random'}",
        f"R {result_bits.get('ret', 0.0):.1f}   step {result_bits.get('step', 0)}   "
        f"ball@piston_{result_bits.get('ball', 0)}",
        f"claimed: {claimed}    selected piston_{state.selected}",
        "A/D select   W/S raise/lower   M claim/release",
        "R reset   T new layout   [ ] physics   L lie   space pause   Q quit",
        "piston_0 is the left wall (goal). + raises, − lowers, release holds.",
    ]
    wanted = result_bits.get("wanted") or []
    for index, agent in enumerate(policy.agents):
        mark = ">" if index == state.selected else " "
        role = "MANUAL" if index in state.manual else "policy"
        wish = f"{wanted[index]:+.2f}" if index < len(wanted) else "  — "
        lines.append(f"{mark} {agent:<10} {role:<7} net {wish}")
    return lines


def _present(
    window: Any,
    frame: np.ndarray,
    lines: list[str],
    *,
    font: Any,
    panel_width: int,
) -> None:
    import pygame

    frame_h, frame_w = frame.shape[:2]
    window.fill((24, 26, 28))
    surface = pygame.image.frombuffer(np.ascontiguousarray(frame).tobytes(), (frame_w, frame_h), "RGB")
    # frombuffer shares memory; copy before the next render reuses the array.
    window.blit(surface.convert(), (0, 0))
    x = frame_w + 12
    y = 8
    for line in lines:
        color = (240, 220, 140) if line.startswith(">") else (230, 230, 230)
        window.blit(font.render(line, True, color), (x, y))
        y += font.get_linesize() + 1
    pygame.display.flip()
    del panel_width


def _open_window(frame: np.ndarray, panel_width: int) -> Any:
    import pygame

    pygame.display.init()
    pygame.font.init()
    height, width = frame.shape[:2]
    window = pygame.display.set_mode((width + panel_width, max(height, 560)))
    pygame.display.set_caption("pistonball diagnostic (CPU)")
    return window


def _poll(state: ControlState, n_pistons: int, n_contexts: int) -> None:
    """Apply discrete keys and store the currently held raise/lower action."""
    import pygame

    state.held_action = 0.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state.quit = True
            continue
        if event.type != pygame.KEYDOWN:
            continue
        key = event.key
        if key in (pygame.K_q, pygame.K_ESCAPE):
            state.quit = True
        elif key in (pygame.K_a, pygame.K_LEFT):
            state.selected = (state.selected - 1) % n_pistons
        elif key in (pygame.K_d, pygame.K_RIGHT):
            state.selected = (state.selected + 1) % n_pistons
        elif key == pygame.K_m:
            if state.selected in state.manual:
                state.manual.discard(state.selected)
            else:
                state.manual.add(state.selected)
        elif key == pygame.K_BACKSPACE:
            state.manual.discard(state.selected)
        elif pygame.K_1 <= key <= pygame.K_9:
            index = key - pygame.K_1
            if index < n_pistons:
                state.selected = index
                state.manual.add(index)
        elif key == pygame.K_r:
            state.reset = True
        elif key == pygame.K_t:
            state.reset = True
            state.new_layout = True
        elif key == pygame.K_LEFTBRACKET:
            state.context_index = (state.context_index - 1) % n_contexts
            state.reset = True
        elif key == pygame.K_RIGHTBRACKET:
            state.context_index = (state.context_index + 1) % n_contexts
            state.reset = True
        elif key == pygame.K_l:
            state.lie = not state.lie
            state.reset = True
        elif key == pygame.K_SPACE:
            state.paused = not state.paused
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_w] or pressed[pygame.K_UP]:
        state.held_action = 1.0
    elif pressed[pygame.K_s] or pressed[pygame.K_DOWN]:
        state.held_action = -1.0


def run_interactive(
    policy: LoadedPolicy,
    env: Any,
    *,
    contexts: list[str],
    start_context: str,
    fixed_drop: bool,
    manual: set[int],
    seed: int,
    log_path: Path | None,
    max_steps: int | None = None,
) -> None:
    """Window loop. ``contexts`` are names ``[`` / ``]`` can switch to."""
    import pygame

    if policy.env_kind != "pistonball":
        raise ValueError(f"pistonball play only supports pistonball checkpoints, got {policy.family}")
    n = len(policy.agents)
    if start_context in contexts:
        context_index = contexts.index(start_context)
    else:
        contexts = [start_context, *contexts]
        context_index = 0
    state = ControlState(manual=set(manual), selected=(sorted(manual)[0] if manual else n // 2))
    state.context_index = context_index
    apply_ctx, apply_after = _context_fns(policy)

    window = None
    font = None
    panel_width = 460
    clock = pygame.time.Clock()

    def begin(ep_seed: int):
        name = contexts[state.context_index]
        lie_name = "default" if state.lie else None
        env_ctx, fed_ctx, env_name, fed_name = prepare_pair(
            policy, name, fixed_drop=fixed_drop, lie_context=lie_name
        )
        apply_ctx(env, env_ctx)
        obs, _ = env.reset(seed=int(ep_seed))
        physics = apply_after(env, env_ctx)
        return obs, physics, env_ctx, fed_ctx, env_name, fed_name

    ep_seed = int(seed)
    obs, physics, _env_ctx, fed_ctx, env_name, fed_name = begin(ep_seed)
    raw = get_raw_pistonball_env(env)
    policy_rows: list[np.ndarray] = []
    applied_rows: list[np.ndarray] = []
    ball_rows: list[int] = []
    ep_return = 0.0
    length = 0
    success = 0.0
    logged = False
    wanted: list[float] = []

    frame = _render_rgb(env)
    if frame is None:
        raise RuntimeError(
            "env.render() returned no frame. On a laptop, run without --script so SDL "
            "is not forced onto the dummy video driver."
        )
    window = _open_window(frame, panel_width)
    font = pygame.font.SysFont("dejavusansmono,monospace", 15)

    def bits() -> dict[str, Any]:
        return {
            "context": env_name,
            "fed": fed_name,
            "fixed_drop": fixed_drop,
            "ret": ep_return,
            "step": length,
            "ball": ball_piston_index(raw) if getattr(raw, "ball", None) is not None else 0,
            "wanted": wanted,
        }

    def flush(truncated: bool) -> None:
        nonlocal logged
        if logged:
            return
        logged = True
        policy_arr = np.stack(policy_rows, axis=0) if policy_rows else np.zeros((0, n))
        applied_arr = np.stack(applied_rows, axis=0) if applied_rows else np.zeros((0, n))
        ball_arr = np.asarray(ball_rows, dtype=np.int32)
        result = EpisodeResult(
            context_name=env_name,
            fed_context_name=fed_name,
            seed=ep_seed,
            manual=sorted(state.manual),
            script=None,
            episode_return=float(ep_return),
            length=int(length),
            success=float(success),
            truncated=truncated,
            physics={k: float(v) for k, v in physics.items()},
            random_drop=bool(getattr(raw, "random_drop", True)),
            policy_mean=_mean_first(policy_arr),
            applied_mean=_mean_first(applied_arr),
            near_obstacle_policy_mean=near_obstacle_policy_mean(policy_arr, ball_arr, set(state.manual)),
            policy_actions=policy_arr,
            applied_actions=applied_arr,
        )
        print(result.format_line(), flush=True)
        if log_path is not None and length > 0:
            append_log(log_path, result, policy=policy)

    print(
        "Window open. piston_0 is the left wall. W raises the selected piston, S lowers it.\n"
        "Claim extra pistons with M or number keys (1 selects piston_0). Unselected claimed\n"
        "pistons stay where they are, so you can park a wall and then drive another piston.",
        flush=True,
    )
    _present(window, frame, _panel_lines(state, bits(), policy), font=font, panel_width=panel_width)

    while not state.quit:
        clock.tick(20)
        _poll(state, n, len(contexts))
        if state.quit:
            flush(truncated=True)
            break
        if state.reset:
            flush(truncated=not state.done)
            if state.new_layout:
                ep_seed += 1
                state.new_layout = False
            state.reset = False
            state.done = False
            state.paused = False
            logged = False
            obs, physics, _env_ctx, fed_ctx, env_name, fed_name = begin(ep_seed)
            raw = get_raw_pistonball_env(env)
            policy_rows = []
            applied_rows = []
            ball_rows = []
            ep_return = 0.0
            length = 0
            success = 0.0
            wanted = []
            frame = _render_rgb(env)
            if frame is not None:
                _present(window, frame, _panel_lines(state, bits(), policy), font=font, panel_width=panel_width)
            continue
        if state.paused or state.done:
            if frame is not None:
                _present(window, frame, _panel_lines(state, bits(), policy), font=font, panel_width=panel_width)
            continue
        if max_steps is not None and length >= int(max_steps):
            state.done = True
            flush(truncated=True)
            continue
        if not env.agents or any(agent not in obs for agent in policy.agents):
            state.done = True
            flush(truncated=False)
            continue

        if state.held_action != 0.0 and state.selected not in state.manual:
            state.manual.add(state.selected)
        with torch.no_grad():
            policy_act = _act(policy, obs, env, fed_ctx)
        wanted = [float(x) for x in np.asarray(policy_act).reshape(n, -1)[:, 0]]
        overrides = {index: 0.0 for index in state.manual}
        if state.selected in state.manual:
            overrides[state.selected] = float(state.held_action)
        applied = apply_overrides(policy_act, overrides)
        policy_rows.append(np.asarray(policy_act, dtype=np.float32).reshape(n, -1)[:, 0])
        applied_rows.append(np.asarray(applied, dtype=np.float32).reshape(n, -1)[:, 0])
        ball_rows.append(ball_piston_index(raw))

        obs, rewards, terms, truncs, _ = env.step(_action_dict(policy, env, applied))
        if rewards:
            ep_return += float(np.mean(list(rewards.values())))
        length += 1
        if terms and any(terms.values()):
            success = 1.0
        frame = _render_rgb(env)
        if frame is not None:
            _present(window, frame, _panel_lines(state, bits(), policy), font=font, panel_width=panel_width)
        ended = (not env.agents) or (
            truncs and all(truncs.get(agent, False) or terms.get(agent, False) for agent in list(truncs))
        )
        if ended:
            state.done = True
            flush(truncated=False)

    pygame.display.quit()


def cycle_names(policy: LoadedPolicy) -> list[str]:
    """Physics names the window can switch to, skipping any this env cannot express."""
    names: list[str] = []
    for name in PHYSICS_CYCLE:
        try:
            resolve_context(policy, name)
        except (KeyError, ValueError):
            continue
        names.append(name)
    return names or ["default"]
