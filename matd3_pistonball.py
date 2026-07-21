"""MATD3 baseline for cooperative multi-agent RL on PettingZoo Pistonball.

This is a *context-free* baseline for contextual-MARL research. It adapts CleanRL's
single-file ``td3_continuous_action.py`` to the multi-agent setting:

* **Multiple actors, one centralized critic (CTDE).** Every piston is controlled by
  its own decentralized actor ``pi_i(o_i)`` (no parameter sharing -> the pure,
  fully-independent baseline). A single *centralized* critic conditions on the joint
  observation and joint action ``Q(o_1..o_N, a_1..a_N)``. Following TD3, the
  centralized critic keeps the twin-Q / target-policy-smoothing / delayed-update
  tricks; "one critic" here means one *centralized* critic (as opposed to one critic
  per agent), which is the standard MATD3 / MADDPG(+TD3) construction. Dropping the
  twin-Q would silently reintroduce the value-overestimation bias TD3 exists to fix,
  so it is retained.
* **No context input.** Actors and the critic never observe the environment context
  (ball physics). This is deliberate: it is the reference point against which
  context-aware algorithms are compared. The context is still *logged and swept* at
  evaluation time to measure zero-shot transfer / generalization.

The environment is ``pistonball_v6`` (fully cooperative, shared reward). Observations
are RGB frames that we grayscale + resize + frame-stack via SuperSuit; each agent
therefore sees a small image and emits a scalar force in ``[-1, 1]``.

Docs for the original algorithm: https://docs.cleanrl.dev/rl-algorithms/td3/
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from pettingzoo.butterfly import pistonball_v6
from torch.utils.tensorboard import SummaryWriter

from evaluation.rliable_utils import save_rliable_scores, summarize_episode_scores


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "contextual-marl"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """if toggled, also save a final model at the end of training (eval checkpoints are separate)"""
    checkpoint_eval: bool = True
    """save a checkpoint after every in-distribution evaluation (keeps only latest + best)"""
    checkpoint_metric: str = "return_mean"
    """eval metric used to decide the best checkpoint (higher is better)"""
    performance_threshold: float | None = None
    """if set, log the first global step whose rolling mean episodic return exceeds this value"""
    rolling_return_window: int = 10
    """window size for the rolling episodic-return mean used in threshold tracking"""
    log_gradient_norms: bool = True
    """log actor/critic gradient norms for optimization-stability analysis"""
    method_tag: str = "hidden_context"
    """label for rliable aggregation (e.g. hidden_context vs context_concat)"""
    reference_return: float = 100.0
    """reference return for optimality gap (Pistonball success-scale heuristic)"""

    # Environment / context arguments. These physics parameters *are* the context in
    # contextual MARL; the baseline trains on the fixed values below and is evaluated
    # zero-shot on perturbed values (see `build_transfer_contexts`).
    n_pistons: int = 20
    """number of pistons == number of agents (each agent controls one piston)"""
    max_cycles: int = 125
    """episode length (truncation horizon) of the environment"""
    time_penalty: float = -0.1
    """per-step reward penalty (context parameter)"""
    ball_mass: float = 0.75
    """training-context ball mass (context parameter)"""
    ball_friction: float = 0.3
    """training-context ball friction (context parameter)"""
    ball_elasticity: float = 1.5
    """training-context ball elasticity (context parameter)"""
    random_drop: bool = True
    """if toggled, the ball spawns at a random x position (context parameter)"""
    random_rotate: bool = True
    """if toggled, the ball spawns with random angular momentum (context parameter)"""

    # Observation preprocessing (SuperSuit).
    frame_size: int = 84
    """observations are resized to (frame_size, frame_size)"""
    frame_stack: int = 3
    """number of grayscale frames stacked together (== input channels)"""

    # Algorithm arguments.
    total_timesteps: int = 1_000_000
    """total environment steps of the experiment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizers"""
    buffer_size: int = 50_000
    """replay buffer capacity in transitions (~0.85 MB/step with 20 agents @ 84px => ~42 GB at 50k; use fewer pistons or lower capacity on small-RAM nodes)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 128
    """the batch size sampled from the replay memory"""
    policy_noise: float = 0.2
    """the scale of target policy smoothing noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise added to actions during rollouts"""
    learning_starts: int = 10_000
    """timestep to start learning (must be < buffer_size so random warmup is retained)"""
    policy_frequency: int = 2
    """the frequency of delayed policy (actor + target) updates"""
    noise_clip: float = 0.5
    """clip range of the target policy smoothing noise"""
    max_grad_norm: float = 10.0
    """clip actor/critic grad norms to this value (0 disables clipping)"""
    feature_dim: int = 128
    """dimensionality of the per-agent CNN feature vector"""
    hidden_dim: int = 256
    """width of the actor/critic MLP heads"""

    # Evaluation / transferability arguments.
    eval_frequency: int = 25_000
    """run an in-distribution evaluation every `eval_frequency` steps (0 disables)"""
    eval_episodes: int = 10
    """number of episodes per evaluation context"""
    transfer_eval: bool = True
    """if toggled, run a zero-shot transfer sweep over perturbed contexts at the end"""
    transfer_factors: tuple[float, ...] = (0.5, 2.0)
    """multiplicative perturbations applied to each physics parameter for transfer eval"""


Context = dict


def get_git_commit() -> str | None:
    """Best-effort git commit hash for reproducibility manifests."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_environment_info() -> dict:
    """Capture runtime and package versions for cluster reproducibility."""
    versions: dict[str, str] = {}
    for package in (
        "torch",
        "numpy",
        "pettingzoo",
        "supersuit",
        "gymnasium",
        "tyro",
        "tensorboard",
        "rliable",
    ):
        try:
            import importlib.metadata as metadata

            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": versions,
        # "git_commit": get_git_commit(),
    }


@dataclass
class RunManifest:
    """Persistent run metadata for cluster bookkeeping and post-hoc analysis."""

    run_name: str
    run_dir: Path
    args: dict
    started_at_unix: float = field(default_factory=time.time)
    environment: dict = field(default_factory=dict)
    global_step: int = 0
    wallclock_seconds: float = 0.0
    best_eval: dict | None = None
    steps_to_threshold: int | None = None
    finished: bool = False

    def write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "started_at_unix": self.started_at_unix,
            "wallclock_seconds": self.wallclock_seconds,
            "global_step": self.global_step,
            "args": self.args,
            "environment": self.environment,
            "best_eval": self.best_eval,
            "steps_to_threshold": self.steps_to_threshold,
            "finished": self.finished,
        }
        with open(self.run_dir / "run_manifest.json", "w") as f:
            json.dump(payload, f, indent=2)


class CheckpointManager:
    """Keep only the latest and best eval checkpoints on disk."""

    def __init__(self, run_dir: Path, metric_key: str = "return_mean"):
        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_key = metric_key
        self.best_value = float("-inf")
        self.best_step = -1
        self.best_stats: dict | None = None

    def save(self, payload: dict, global_step: int, eval_stats: dict) -> dict[str, str | float]:
        metric = float(eval_stats[self.metric_key])
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(
            {
                **payload,
                "global_step": global_step,
                "eval_stats": eval_stats,
                "checkpoint_metric": self.metric_key,
                "checkpoint_metric_value": metric,
            },
            latest_path,
        )
        saved: dict[str, str | float] = {"latest": str(latest_path), "metric": metric}

        if metric > self.best_value:
            self.best_value = metric
            self.best_step = global_step
            self.best_stats = dict(eval_stats)
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(
                {
                    **payload,
                    "global_step": global_step,
                    "eval_stats": eval_stats,
                    "checkpoint_metric": self.metric_key,
                    "checkpoint_metric_value": metric,
                },
                best_path,
            )
            saved["best"] = str(best_path)
            saved["best_step"] = global_step
        return saved


def build_checkpoint_payload(nets: dict, args: Args) -> dict:
    return {
        "actors": nets["actors"].state_dict(),
        "target_actors": nets["target_actors"].state_dict(),
        "qf1": nets["qf1"].state_dict(),
        "qf2": nets["qf2"].state_dict(),
        "qf1_target": nets["qf1_target"].state_dict(),
        "qf2_target": nets["qf2_target"].state_dict(),
        "q_optimizer": nets["q_optimizer"].state_dict(),
        "actor_optimizer": nets["actor_optimizer"].state_dict(),
        "args": vars(args),
    }


def append_training_episode_record(
    run_dir: Path,
    global_step: int,
    ep_return: float,
    ep_len: int,
    success: bool,
    per_agent_return_variance: float,
) -> None:
    """Append team-level training episodes for learning-curve / seed aggregation."""
    record = {
        "global_step": global_step,
        "team_episodic_return": ep_return,
        "episodic_length": ep_len,
        "success": success,
        "per_agent_return_variance": per_agent_return_variance,
    }
    with open(run_dir / "training_episodes.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def log_eval_metrics(writer: SummaryWriter, prefix: str, stats: dict, global_step: int) -> None:
    """Log metrics using abstract §Evaluation taxonomy (performance / coordination / …)."""
    mapping = {
        "return_mean": f"{prefix}/performance/team_return_mean",
        "return_std": f"{prefix}/performance/team_return_std",
        "return_median": f"{prefix}/performance/team_return_median",
        "return_iqm": f"{prefix}/performance/team_return_iqm",
        "optimality_gap": f"{prefix}/performance/optimality_gap",
        "length_mean": f"{prefix}/performance/episode_length_mean",
        "success_rate": f"{prefix}/performance/success_rate",
        "per_agent_return_variance_mean": f"{prefix}/coordination/per_agent_return_variance",
        "action_std_mean": f"{prefix}/coordination/action_std_across_agents",
    }
    for key, tag in mapping.items():
        if key in stats:
            writer.add_scalar(tag, stats[key], global_step)


def append_eval_record(run_dir: Path, global_step: int, eval_stats: dict, wallclock_seconds: float) -> None:
    record = {
        "global_step": global_step,
        "wallclock_seconds": wallclock_seconds,
        **eval_stats,
    }
    with open(run_dir / "eval_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def log_timing_metrics(
    writer: SummaryWriter,
    global_step: int,
    start_time: float,
    total_timesteps: int,
) -> None:
    elapsed = time.time() - start_time
    writer.add_scalar("time/wallclock_seconds", elapsed, global_step)
    if global_step > 0:
        sps = global_step / elapsed
        writer.add_scalar("charts/SPS", int(sps), global_step)
        remaining = max(total_timesteps - global_step, 0)
        writer.add_scalar("time/eta_hours", (remaining / sps) / 3600.0, global_step)
        writer.add_scalar("time/progress", global_step / total_timesteps, global_step)


def gradient_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def clip_gradients(parameters, max_grad_norm: float) -> None:
    """Clip gradients in-place when ``max_grad_norm > 0``; no-op otherwise."""
    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters, max_grad_norm)


def bootstrap_done(terminated: bool, truncated: bool) -> float:
    """Mark episode end for the TD target.

    Pistonball's ``max_cycles`` is a true episodic horizon (not an artificial
    TimeLimit around a continuing MDP), so both success termination and timeout
    truncation must stop bootstrapping. Otherwise every timeout incorrectly
    continues with ``gamma * Q(s')``.
    """
    return float(terminated or truncated)


def default_context(args: Args) -> Context:
    """The training context (the fixed environment the baseline is trained on)."""
    return {
        "time_penalty": args.time_penalty,
        "ball_mass": args.ball_mass,
        "ball_friction": args.ball_friction,
        "ball_elasticity": args.ball_elasticity,
        "random_drop": args.random_drop,
        "random_rotate": args.random_rotate,
    }


def build_transfer_contexts(args: Args) -> dict[str, Context]:
    """Build the zero-shot transfer sweep.

    Returns a mapping ``name -> context``, including the ``"train"`` context and one
    perturbed context per (physics parameter, factor) pair. Only the continuous
    physics parameters are swept, since these define a well-ordered context space in
    which generalization can be measured.
    """
    base = default_context(args)
    contexts: dict[str, Context] = {"train": base}
    for param in ("ball_mass", "ball_friction", "ball_elasticity"):
        for factor in args.transfer_factors:
            ctx = dict(base)
            ctx[param] = base[param] * factor
            contexts[f"{param}_x{factor:g}"] = ctx
    return contexts


def make_env(args: Args, context: Context, render_mode: str | None = None):
    """Create a preprocessed Pistonball parallel environment for a given context."""
    env = pistonball_v6.parallel_env(
        n_pistons=args.n_pistons,
        continuous=True,
        max_cycles=args.max_cycles,
        time_penalty=context["time_penalty"],
        ball_mass=context["ball_mass"],
        ball_friction=context["ball_friction"],
        ball_elasticity=context["ball_elasticity"],
        random_drop=context["random_drop"],
        random_rotate=context["random_rotate"],
        render_mode=render_mode,
    )
    # Grayscale -> resize -> frame-stack. After this each agent observes an image of
    # shape (frame_size, frame_size, frame_stack).
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=args.frame_size, y_size=args.frame_size)
    env = ss.frame_stack_v1(env, args.frame_stack)
    return env


def stack_obs(obs_dict: dict, agents: list[str]) -> np.ndarray:
    """Stack a per-agent obs dict into a channel-first array ``(N, C, H, W)`` uint8."""
    frames = np.stack([obs_dict[a] for a in agents], axis=0)  # (N, H, W, C)
    return np.transpose(frames, (0, 3, 1, 2)).copy()  # (N, C, H, W)


class MultiAgentReplayBuffer:
    """Uint8 image replay buffer for a fully-cooperative (shared-reward) MARL task.

    Observations dominate memory, so they are stored as ``uint8`` and normalized to
    ``[0, 1]`` only when a batch is sampled. Because Pistonball gives every agent the
    same scalar reward and terminates all agents simultaneously, we store a single
    reward and a single done flag per transition instead of per-agent copies.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_shape: tuple[int, int, int],
        act_dim: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.full = False

        self.obs = np.zeros((capacity, n_agents, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, n_agents, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        reward: float,
        done: float,
    ) -> None:
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int):
        high = len(self)
        idx = np.random.randint(0, high, size=batch_size)
        to = lambda a: torch.as_tensor(a, device=self.device)
        obs = to(self.obs[idx]).float() / 255.0
        next_obs = to(self.next_obs[idx]).float() / 255.0
        actions = to(self.actions[idx])
        rewards = to(self.rewards[idx])
        dones = to(self.dones[idx])
        return obs, next_obs, actions, rewards, dones

    def memory_gb(self) -> float:
        arrays = (self.obs, self.next_obs, self.actions, self.rewards, self.dones)
        return sum(a.nbytes for a in arrays) / 1e9


def estimate_buffer_gb(n_agents: int, obs_shape: tuple[int, int, int], capacity: int) -> float:
    """Rough replay-buffer footprint in GB (uint8 obs + float32 actions/rewards)."""
    c, h, w = obs_shape
    obs_bytes = n_agents * c * h * w
    per_transition = 2 * obs_bytes + n_agents * 4 + 8  # obs + next_obs + actions + reward + done
    return capacity * per_transition / 1e9


class CNNEncoder(nn.Module):
    """Small CNN mapping an image to a feature vector.

    Uses stride-2 convolutions followed by adaptive pooling so it works for any
    ``frame_size`` (the flatten dimension is fixed regardless of input resolution),
    which keeps the network configurable and the tests cheap.
    """

    def __init__(self, in_channels: int, frame_size: int, feature_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, frame_size, frame_size)).shape[1]
        self.fc = nn.Linear(n_flat, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self.conv(x)))


class Actor(nn.Module):
    """Decentralized actor: maps a single agent's observation to its action."""

    def __init__(self, args: Args, act_dim: int, action_low: float, action_high: float):
        super().__init__()
        self.encoder = CNNEncoder(args.frame_stack, args.frame_size, args.feature_dim)
        self.fc1 = nn.Linear(args.feature_dim, args.hidden_dim)
        self.fc_mu = nn.Linear(args.hidden_dim, act_dim)
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(self.encoder(obs)))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias


class CentralizedCritic(nn.Module):
    """Centralized Q-network over the joint observation and joint action.

    Each agent's image is encoded by a shared CNN (the agents are homogeneous, so
    weight sharing in the *critic's* encoder is both correct and efficient), the
    resulting features are concatenated with all agents' actions, and an MLP maps the
    joint representation to a single scalar team value.
    """

    def __init__(self, args: Args, n_agents: int, act_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.encoder = CNNEncoder(args.frame_stack, args.frame_size, args.feature_dim)
        joint_dim = n_agents * (args.feature_dim + act_dim)
        self.fc1 = nn.Linear(joint_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # obs: (B, N, C, H, W), actions: (B, N, act_dim)
        b, n = obs.shape[:2]
        feats = self.encoder(obs.reshape(b * n, *obs.shape[2:])).reshape(b, -1)
        x = torch.cat([feats, actions.reshape(b, -1)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def joint_policy_actions(actors: nn.ModuleList, obs: torch.Tensor) -> torch.Tensor:
    """Stack per-agent actor outputs into a joint action ``(B, N, act_dim)``."""
    return torch.stack([actor(obs[:, i]) for i, actor in enumerate(actors)], dim=1)


@torch.no_grad()
def evaluate(
    actors: nn.ModuleList,
    args: Args,
    context: Context,
    agents: list[str],
    device: torch.device,
    seed: int,
    n_episodes: int,
    render_mode: str | None = None,
):
    """Roll out the deterministic joint policy in ``context`` and summarize returns.

    Returns a dict with mean/std episodic return, mean length and success rate (the
    fraction of episodes ending because the ball reached the wall). Actions are
    deterministic (no exploration noise) so this measures the learned policy directly.
    """
    env = make_env(args, context, render_mode=render_mode)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[float] = []
    episode_action_stds: list[float] = []
    episode_per_agent_var: list[float] = []

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        ep_return, ep_len, success = 0.0, 0, 0.0
        ep_action_stds: list[float] = []
        ep_per_agent_vars: list[float] = []
        while env.agents:
            obs = torch.as_tensor(stack_obs(obs_dict, agents), device=device).float() / 255.0
            actions = joint_policy_actions(actors, obs.unsqueeze(0)).squeeze(0).cpu().numpy()
            ep_action_stds.append(float(np.std(actions)))
            action_dict = {a: actions[i] for i, a in enumerate(agents)}
            obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
            rew_vals = list(rewards.values())
            ep_return += float(np.mean(rew_vals))
            ep_per_agent_vars.append(float(np.var(rew_vals)))
            ep_len += 1
            if any(terms.values()):
                success = 1.0
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_successes.append(success)
        episode_action_stds.append(float(np.mean(ep_action_stds)) if ep_action_stds else 0.0)
        episode_per_agent_var.append(float(np.mean(ep_per_agent_vars)) if ep_per_agent_vars else 0.0)
    env.close()

    summary = summarize_episode_scores(episode_returns, reference_return=args.reference_return)
    return {
        **summary,
        "length_mean": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(episode_successes)),
        "action_std_mean": float(np.mean(episode_action_stds)),
        "per_agent_return_variance_mean": float(np.mean(episode_per_agent_var)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
    }


def build_agents(args: Args, device: torch.device):
    """Construct the environment metadata and all networks/optimizers.

    Returned as a plain dict so the training loop and the tests can share the exact
    same construction logic.
    """
    probe_env = make_env(args, default_context(args))
    probe_env.reset(seed=args.seed)
    agents = list(probe_env.possible_agents)
    n_agents = len(agents)
    act_space = probe_env.action_space(agents[0])
    act_dim = int(np.prod(act_space.shape))
    action_low = float(act_space.low[0])
    action_high = float(act_space.high[0])
    obs_shape = (args.frame_stack, args.frame_size, args.frame_size)
    probe_env.close()

    def make_actor():
        return Actor(args, act_dim, action_low, action_high).to(device)

    actors = nn.ModuleList([make_actor() for _ in range(n_agents)])
    target_actors = nn.ModuleList([make_actor() for _ in range(n_agents)])
    qf1 = CentralizedCritic(args, n_agents, act_dim).to(device)
    qf2 = CentralizedCritic(args, n_agents, act_dim).to(device)
    qf1_target = CentralizedCritic(args, n_agents, act_dim).to(device)
    qf2_target = CentralizedCritic(args, n_agents, act_dim).to(device)
    target_actors.load_state_dict(actors.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actors.parameters(), lr=args.learning_rate)

    return {
        "agents": agents,
        "n_agents": n_agents,
        "act_dim": act_dim,
        "action_low": action_low,
        "action_high": action_high,
        "obs_shape": obs_shape,
        "actors": actors,
        "target_actors": target_actors,
        "qf1": qf1,
        "qf2": qf2,
        "qf1_target": qf1_target,
        "qf2_target": qf2_target,
        "q_optimizer": q_optimizer,
        "actor_optimizer": actor_optimizer,
    }


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1 - tau).add_(tau * param.data)


def train(args: Args) -> str:
    run_name = f"pistonball__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_name=run_name,
        run_dir=run_dir,
        args=vars(args),
        environment=collect_environment_info(),
    )
    manifest.write()
    checkpoint_manager = (
        CheckpointManager(run_dir, metric_key=args.checkpoint_metric) if args.checkpoint_eval else None
    )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    nets = build_agents(args, device)
    agents = nets["agents"]
    actors, target_actors = nets["actors"], nets["target_actors"]
    qf1, qf2 = nets["qf1"], nets["qf2"]
    qf1_target, qf2_target = nets["qf1_target"], nets["qf2_target"]
    q_optimizer, actor_optimizer = nets["q_optimizer"], nets["actor_optimizer"]
    action_low, action_high = nets["action_low"], nets["action_high"]
    action_scale = actors[0].action_scale

    rb = MultiAgentReplayBuffer(
        args.buffer_size, nets["n_agents"], nets["obs_shape"], nets["act_dim"], device
    )
    buf_gb = rb.memory_gb()
    print(f"replay buffer allocation: {buf_gb:.2f} GB ({args.buffer_size} transitions)")
    if args.learning_starts >= args.buffer_size:
        raise ValueError(
            f"learning_starts ({args.learning_starts}) must be < buffer_size ({args.buffer_size}) "
            "so the random-exploration warmup is not overwritten before learning begins."
        )
    # Host-RAM guard: uint8 obs live on CPU. ~42 GB at defaults (20 agents, 84px, 50k).
    if buf_gb > 48.0:
        raise ValueError(
            f"Replay buffer would use {buf_gb:.1f} GB. Reduce --buffer-size, --n-pistons, "
            f"--frame-size, or --frame-stack. At defaults (20 agents, 84px, 50k steps) expect ~42 GB."
        )

    env = make_env(args, default_context(args))
    obs_dict, _ = env.reset(seed=args.seed)
    obs = stack_obs(obs_dict, agents)
    ep_return, ep_len = 0.0, 0
    ep_per_agent_var_sum, ep_var_steps = 0.0, 0
    start_time = time.time()
    recent_returns: deque[float] = deque(maxlen=args.rolling_return_window)
    steps_to_threshold: int | None = None
    actor_loss = torch.tensor(0.0)

    for global_step in range(args.total_timesteps):
        # --- action selection ---------------------------------------------------
        if global_step < args.learning_starts:
            actions = np.random.uniform(
                action_low, action_high, size=(nets["n_agents"], nets["act_dim"])
            ).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).float().unsqueeze(0) / 255.0
                actions = joint_policy_actions(actors, obs_t).squeeze(0)
                actions += torch.normal(0, action_scale * args.exploration_noise, size=actions.shape, device=device)
                actions = actions.cpu().numpy().clip(action_low, action_high).astype(np.float32)

        # --- environment step ---------------------------------------------------
        action_dict = {a: actions[i] for i, a in enumerate(agents)}
        next_obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
        reward = float(np.mean(list(rewards.values())))
        rew_vals = list(rewards.values())
        ep_per_agent_var_sum += float(np.var(rew_vals))
        ep_var_steps += 1
        terminated = any(terms.values())
        truncated = any(truncs.values())
        next_obs = stack_obs(next_obs_dict, agents)

        ep_return += reward
        ep_len += 1

        # Stop bootstrapping on both success and max_cycles timeout (true episode end).
        rb.add(obs, next_obs, actions, reward, bootstrap_done(terminated, truncated))
        obs = next_obs

        if terminated or truncated:
            ep_var_mean = ep_per_agent_var_sum / max(ep_var_steps, 1)
            writer.add_scalar("performance/team_episodic_return", ep_return, global_step)
            writer.add_scalar("performance/episodic_length", ep_len, global_step)
            writer.add_scalar("performance/success", float(terminated), global_step)
            writer.add_scalar("coordination/per_agent_return_variance", ep_var_mean, global_step)
            append_training_episode_record(
                run_dir, global_step, ep_return, ep_len, bool(terminated), ep_var_mean
            )
            recent_returns.append(ep_return)
            if len(recent_returns) == recent_returns.maxlen:
                rolling_mean = float(np.mean(recent_returns))
                writer.add_scalar("sample_efficiency/rolling_return_mean", rolling_mean, global_step)
                if (
                    args.performance_threshold is not None
                    and steps_to_threshold is None
                    and rolling_mean >= args.performance_threshold
                ):
                    steps_to_threshold = global_step
                    writer.add_scalar(
                        "sample_efficiency/steps_to_threshold", steps_to_threshold, global_step
                    )
                    manifest.steps_to_threshold = steps_to_threshold
            print(f"global_step={global_step}, episodic_return={ep_return:.2f}, length={ep_len}")
            obs_dict, _ = env.reset(seed=args.seed + global_step)
            obs = stack_obs(obs_dict, agents)
            ep_return, ep_len = 0.0, 0
            ep_per_agent_var_sum, ep_var_steps = 0.0, 0

        # --- learning -----------------------------------------------------------
        if global_step > args.learning_starts:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)

            with torch.no_grad():
                next_actions = []
                for i, target_actor in enumerate(target_actors):
                    noise = (torch.randn_like(b_actions[:, i]) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip
                    ) * action_scale
                    a = (target_actor(b_next_obs[:, i]) + noise).clamp(action_low, action_high)
                    next_actions.append(a)
                next_actions = torch.stack(next_actions, dim=1)

                qf1_next = qf1_target(b_next_obs, next_actions)
                qf2_next = qf2_target(b_next_obs, next_actions)
                min_q_next = torch.min(qf1_next, qf2_next)
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * min_q_next

            qf1_a = qf1(b_obs, b_actions)
            qf2_a = qf2(b_obs, b_actions)
            qf1_loss = F.mse_loss(qf1_a, next_q_value)
            qf2_loss = F.mse_loss(qf2_a, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_params = list(qf1.parameters()) + list(qf2.parameters())
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_grad_norm = gradient_norm(q_params)
            clip_gradients(q_params, args.max_grad_norm)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(b_obs, joint_policy_actions(actors, b_obs)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = gradient_norm(actors.parameters())
                clip_gradients(actors.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                polyak_update(actors, target_actors, args.tau)
                polyak_update(qf1, qf1_target, args.tau)
                polyak_update(qf2, qf2_target, args.tau)
            else:
                actor_grad_norm = 0.0

            if global_step % 100 == 0:
                log_timing_metrics(writer, global_step, start_time, args.total_timesteps)
                writer.add_scalar("losses/qf1_values", qf1_a.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                if args.log_gradient_norms:
                    writer.add_scalar("stability/q_grad_norm", q_grad_norm, global_step)
                    writer.add_scalar("stability/actor_grad_norm", actor_grad_norm, global_step)
                elapsed = time.time() - start_time
                print(f"SPS: {int(global_step / elapsed)} | wallclock: {elapsed / 3600:.2f} h")

        # --- periodic in-distribution evaluation --------------------------------
        if (
            args.eval_frequency > 0
            and global_step > args.learning_starts
            and global_step % args.eval_frequency == 0
        ):
            eval_start = time.time()
            stats = evaluate(
                actors, args, default_context(args), agents, device,
                seed=args.seed + 10_000, n_episodes=args.eval_episodes,
            )
            eval_wallclock = time.time() - eval_start
            total_wallclock = time.time() - start_time
            stats["eval_wallclock_seconds"] = eval_wallclock
            stats["total_wallclock_seconds"] = total_wallclock
            log_eval_metrics(writer, "eval_id", stats, global_step)
            writer.add_scalar("time/eval_wallclock_seconds", eval_wallclock, global_step)
            writer.add_scalar("time/total_wallclock_seconds", total_wallclock, global_step)
            append_eval_record(run_dir, global_step, stats, total_wallclock)
            save_rliable_scores(
                run_dir,
                algorithm=args.exp_name,
                method_tag=args.method_tag,
                seed=args.seed,
                global_step=global_step,
                context_order=["train"],
                scores_by_context={"train": stats["episode_returns"]},
                eval_split="in_distribution_eval",
                reference_return=args.reference_return,
                extra={"checkpoint_metric": args.checkpoint_metric},
            )
            print(
                f"[eval @ {global_step}] return={stats['return_mean']:.2f} "
                f"success={stats['success_rate']:.2f} "
                f"eval_time={eval_wallclock:.1f}s total_time={total_wallclock / 3600:.2f}h"
            )
            if checkpoint_manager is not None:
                saved = checkpoint_manager.save(
                    build_checkpoint_payload(nets, args),
                    global_step,
                    stats,
                )
                print(f"[checkpoint] latest={saved['latest']}" + (f" best={saved['best']}" if "best" in saved else ""))
                manifest.best_eval = {
                    "step": checkpoint_manager.best_step,
                    "metric": args.checkpoint_metric,
                    "value": checkpoint_manager.best_value,
                    "stats": checkpoint_manager.best_stats,
                }
            else:
                manifest.best_eval = {
                    "step": global_step,
                    "metric": args.checkpoint_metric,
                    "value": stats[args.checkpoint_metric],
                    "stats": stats,
                }
            manifest.global_step = global_step
            manifest.wallclock_seconds = total_wallclock
            manifest.write()
    env.close()

    # --- final transfer / generalization sweep ---------------------------------
    if args.transfer_eval:
        transfer_start = time.time()
        results = transfer_sweep(actors, args, agents, device, run_dir=run_dir)
        results["transfer_wallclock_seconds"] = time.time() - transfer_start
        results["total_wallclock_seconds"] = time.time() - start_time
        with open(run_dir / "transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)
        log_transfer_results(writer, results, args.total_timesteps)
        print("transfer sweep:", json.dumps(results["contexts"], indent=2))
        print(f"generalization gap: {results['generalization_gap']:.3f}")

    if args.save_model:
        model_path = run_dir / f"{args.exp_name}_final.pt"
        torch.save(build_checkpoint_payload(nets, args), model_path)
        print(f"model saved to {model_path}")

    manifest.global_step = args.total_timesteps
    manifest.wallclock_seconds = time.time() - start_time
    manifest.finished = True
    manifest.write()
    print(f"run finished in {manifest.wallclock_seconds / 3600:.2f} h -> {run_dir}")

    writer.close()
    return run_name


def transfer_sweep(
    actors: nn.ModuleList,
    args: Args,
    agents: list[str],
    device: torch.device,
    run_dir: Path | None = None,
):
    """Evaluate the frozen policy across the transfer contexts and summarize.

    Reports generalization gap (ID vs OOD) using team return means and IQMs, and
    writes ``rliable_scores.json`` for multi-seed robust aggregation.
    """
    contexts = build_transfer_contexts(args)
    context_order = list(contexts.keys())
    per_context: dict[str, dict] = {}
    scores_by_context: dict[str, list[float]] = {}
    for name, ctx in contexts.items():
        stats = evaluate(
            actors, args, ctx, agents, device,
            seed=args.seed + 20_000, n_episodes=args.eval_episodes,
        )
        stats["context"] = ctx
        per_context[name] = stats
        scores_by_context[name] = stats["episode_returns"]

    train_stats = per_context["train"]
    ood_names = [k for k in per_context if k != "train"]
    ood_means = [per_context[k]["return_mean"] for k in ood_names]
    ood_iqms = [per_context[k]["return_iqm"] for k in ood_names]

    results = {
        "contexts": per_context,
        "train_return_mean": train_stats["return_mean"],
        "train_return_iqm": train_stats["return_iqm"],
        "ood_return_mean": float(np.mean(ood_means)) if ood_means else train_stats["return_mean"],
        "ood_return_iqm": float(np.mean(ood_iqms)) if ood_iqms else train_stats["return_iqm"],
        "generalization_gap_mean": train_stats["return_mean"] - (float(np.mean(ood_means)) if ood_means else train_stats["return_mean"]),
        "generalization_gap_iqm": train_stats["return_iqm"] - (float(np.mean(ood_iqms)) if ood_iqms else train_stats["return_iqm"]),
        # Backwards-compatible aliases
        "train_return": train_stats["return_mean"],
        "ood_return_mean_legacy": float(np.mean(ood_means)) if ood_means else train_stats["return_mean"],
        "generalization_gap": train_stats["return_mean"] - (float(np.mean(ood_means)) if ood_means else train_stats["return_mean"]),
    }

    if run_dir is not None:
        save_rliable_scores(
            run_dir,
            algorithm=args.exp_name,
            method_tag=args.method_tag,
            seed=args.seed,
            global_step=args.total_timesteps,
            context_order=context_order,
            scores_by_context=scores_by_context,
            eval_split="zero_shot_transfer",
            reference_return=args.reference_return,
            extra={
                "generalization_gap_mean": results["generalization_gap_mean"],
                "generalization_gap_iqm": results["generalization_gap_iqm"],
            },
        )
    return results


def log_transfer_results(writer: SummaryWriter, results: dict, global_step: int) -> None:
    for name, stats in results["contexts"].items():
        log_eval_metrics(writer, f"transfer/{name}", stats, global_step)
    writer.add_scalar("generalization/id_return_mean", results["train_return_mean"], global_step)
    writer.add_scalar("generalization/ood_return_mean", results["ood_return_mean"], global_step)
    writer.add_scalar("generalization/gap_mean", results["generalization_gap_mean"], global_step)
    writer.add_scalar("generalization/id_return_iqm", results["train_return_iqm"], global_step)
    writer.add_scalar("generalization/ood_return_iqm", results["ood_return_iqm"], global_step)
    writer.add_scalar("generalization/gap_iqm", results["generalization_gap_iqm"], global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
