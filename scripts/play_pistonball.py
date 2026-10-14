#!/usr/bin/env python3
"""Drive some Pistonball pistons yourself and leave the rest to a checkpoint.

CPU is enough: pygame is software-rendered, and each step is one forward pass
of the saved actor. Do not pass ``--cuda`` on a machine without a GPU.

Interactive (laptop, a display is required)::

    python scripts/play_pistonball.py \\
        --exp-name baseline5_varying_randdrop --seed 3 --human center

Scripted, including on the cluster (no window)::

    python scripts/play_pistonball.py \\
        --exp-name baseline5_varying_randdrop --seed 1 \\
        --script block --human 2 --episode-seed 0 --max-steps 40

    python scripts/play_pistonball.py --suggest

``piston_0`` is the left wall, which is the goal. W raises, S lowers, and
releasing the key holds the piston where it is. See ``--suggest`` for which
runs are worth playing and which interventions to compare.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_ARGV = sys.argv[1:]


def _flag_present(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


# rollout_viz forces a dummy SDL driver if this is still unset at import.
# Pin a real driver first when a window is actually wanted.
_WINDOW = (
    not _flag_present(_ARGV, "--script")
    and not _flag_present(_ARGV, "--headless")
    and not _flag_present(_ARGV, "--suggest")
    and bool(os.environ.get("DISPLAY"))
)
if _WINDOW:
    if not os.environ.get("SDL_VIDEODRIVER"):
        os.environ["SDL_VIDEODRIVER"] = {"Windows": "windows", "Darwin": "cocoa"}.get(
            platform.system(), "x11"
        )
else:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import tyro

from evaluation.pistonball_play import (
    SCRIPTS,
    SUGGESTED_SESSIONS,
    append_log,
    cycle_names,
    make_play_env,
    parse_piston_spec,
    prepare_pair,
    run_interactive,
    run_scripted_episode,
    save_gif,
)
from evaluation.rollout_viz import load_policy, resolve_run_dirs


@dataclass
class Args:
    runs_root: str = "runs"
    """Root passed to ``--exp-name``."""

    exp_name: str | None = None
    """Experiment name or preset, for example ``baseline5_varying_randdrop``."""

    run_dir: str | None = None
    """One run directory containing ``checkpoints/`` and ``run_manifest.json``."""

    seed: int | None = None
    """Training seed. Required when ``--exp-name`` matches more than one run."""

    checkpoint: str = "best"
    """``best`` or ``latest``."""

    context: str = "default"
    """Starting physics. Names like ``ball_mass_x4`` or ``gravity_x0.25`` work."""

    human: str = "center"
    """Pistons you control: ``center``, ``left``, ``right``, ``2``, or ``1,3``."""

    script: str | None = None
    """No window. One of policy, assist, block, pit, freeze."""

    headless: bool = False
    """Force the dummy video driver. Implied by ``--script``."""

    lie_context: str | None = None
    """Context vector fed to a concat actor while the env uses ``--context``."""

    fixed_drop: bool = True
    """Spawn the ball on the right with no random spin, so repeats match."""

    episode_seed: int = 0
    """Reset seed. Same value → same initial piston heights."""

    episodes: int = 1
    """Scripted repeats. The window ignores this and runs until you quit."""

    max_steps: int | None = None
    """Cut an episode early. Default is the checkpoint's ``max_cycles``."""

    save_gif: str | None = None
    """Write a GIF of the last scripted episode."""

    log: bool = True
    """Append a JSON line per episode under ``reports/play_logs/``."""

    suggest: bool = False
    """Print the short list of runs and interventions, then exit."""

    cuda: bool = False
    """Load the actor on CUDA. Leave this off on a CPU-only machine."""


def _resolve_run(args: Args) -> Path:
    if args.run_dir:
        path = Path(args.run_dir)
        if not (path / "run_manifest.json").exists():
            raise SystemExit(f"no run_manifest.json in {path}")
        return path
    if not args.exp_name:
        raise SystemExit("pass --exp-name or --run-dir (or --suggest)")
    found = resolve_run_dirs(args.runs_root, args.exp_name, seeds=args.seed)
    if args.seed is None and len(found) > 1:
        seeds = []
        for path in found:
            # run folder names end with __seedN__timestamp
            parts = path.name.split("__")
            label = next((p for p in parts if p.startswith("seed")), path.name)
            seeds.append(label)
        raise SystemExit(
            f"{args.exp_name} has {len(found)} runs ({', '.join(seeds)}). Pass --seed."
        )
    return found[0]


def main(args: Args) -> None:
    if args.suggest:
        print(SUGGESTED_SESSIONS)
        return
    if args.script is not None and args.script not in SCRIPTS:
        raise SystemExit(f"--script must be one of: {', '.join(SCRIPTS)}")
    interactive = args.script is None
    if interactive and not os.environ.get("DISPLAY"):
        raise SystemExit(
            "No DISPLAY, so pygame cannot open a window on this machine.\n"
            "Run the window on your CPU laptop. On the cluster, check the same\n"
            "control logic without a display, for example:\n"
            "  python scripts/play_pistonball.py --exp-name baseline5_varying_randdrop "
            "--seed 1 --script block --human 2 --max-steps 30"
        )

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.cuda and device.type != "cuda":
        print("CUDA was requested but is not available; using CPU.", flush=True)
    run_dir = _resolve_run(args)
    policy = load_policy(run_dir, checkpoint=args.checkpoint, device=device)
    if policy.env_kind != "pistonball":
        raise SystemExit(f"{run_dir.name} is {policy.family}, not pistonball")
    n = len(policy.agents)
    manual = parse_piston_spec(args.human, n)
    print(
        f"device: {device.type}  (pygame and this actor do not need a GPU)\n"
        f"loaded: {policy.label}\n"
        f"from:   {policy.run_dir}\n"
        f"pistons: {n}   you control: {manual or 'none'}   "
        f"piston_0 is the left wall / goal",
        flush=True,
    )

    env_ctx, fed_ctx, env_name, fed_name = prepare_pair(
        policy, args.context, fixed_drop=args.fixed_drop, lie_context=args.lie_context
    )
    env = make_play_env(policy, env_ctx)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None
    if args.log:
        safe = policy.exp_name.replace("/", "_")
        log_path = _ROOT / "reports" / "play_logs" / f"{safe}_seed{policy.seed}_{stamp}.jsonl"

    try:
        if interactive:
            names = cycle_names(policy)
            run_interactive(
                policy,
                env,
                contexts=names,
                start_context=args.context,
                fixed_drop=args.fixed_drop,
                manual=set(manual),
                seed=args.episode_seed,
                log_path=log_path,
                max_steps=args.max_steps,
            )
            return

        assert args.script is not None
        returns: list[float] = []
        frames: list = []
        for episode in range(int(args.episodes)):
            ep_seed = int(args.episode_seed) + episode
            result, frames = run_scripted_episode(
                policy,
                env,
                env_ctx,
                fed_ctx,
                context_name=env_name,
                fed_context_name=fed_name,
                seed=ep_seed,
                manual=set(manual),
                script=args.script,
                max_steps=args.max_steps,
                collect_frames=bool(args.save_gif) and episode == int(args.episodes) - 1,
            )
            returns.append(result.episode_return)
            print(result.format_line(), flush=True)
            if log_path is not None:
                append_log(log_path, result, policy=policy)
        if len(returns) > 1:
            print(f"mean return over {len(returns)} episodes: {sum(returns) / len(returns):.1f}", flush=True)
        if args.save_gif:
            save_gif(Path(args.save_gif), frames)
            print(f"wrote {args.save_gif}", flush=True)
        if log_path is not None:
            print(f"log: {log_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
