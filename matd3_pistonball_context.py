"""Context-aware MATD3 for cooperative Pistonball (contextual MARL).

Sibling of ``matd3_pistonball.py`` (hidden-context / context-free baseline). This
script feeds an explicit environment-context vector into the actors and/or the
centralized critic. Context inclusion is pluggable via ``--context-mode``:

* ``concat`` (default / simplest): concatenate a normalized continuous context
  vector with the CNN features (actor) and with the joint features (critic).
* ``none``: ignore context inputs (parity check against the hidden baseline;
  useful for ablation / plumbing tests).

Future modes (FiLM, hypernetworks, …) can be added behind the same
``context_mode`` switch without changing the training loop contract.

The continuous context parameters are ``time_penalty``, ``ball_mass``,
``ball_friction``, and ``ball_elasticity``. Boolean spawn flags remain env-only
(they are not numeric features). Vectors are normalized by the *training*
context magnitudes so the train vector is near unit scale and OOD factors map
to interpretable magnitudes (e.g. ``ball_mass_x2`` -> ~2).

At transfer evaluation the *true* context of each evaluation environment is
provided to the networks (oracle / privileged context), which is the standard
contextual-RL comparison against a hidden-context policy.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import matd3_pistonball as base
from evaluation.rliable_utils import save_rliable_scores, summarize_episode_scores

Context = base.Context
ContextMode = Literal["concat", "none"]

# Continuous physics/context features exposed to the networks.
CONTEXT_KEYS: tuple[str, ...] = (
    "time_penalty",
    "ball_mass",
    "ball_friction",
    "ball_elasticity",
)


@dataclass
class Args(base.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """algorithm / script name (kept for manifests and rliable aggregation)"""
    experiment: str = "context_concat"
    """human-readable experiment tag used in run folder names"""
    method_tag: str = "context_concat"
    """label for rliable aggregation (e.g. context_concat vs hidden_context)"""
    context_mode: ContextMode = "concat"
    """how context is injected into actor/critic (concat | none)"""
    context_to_actor: bool = True
    """if toggled, feed context into decentralized actors"""
    context_to_critic: bool = True
    """if toggled, feed context into the centralized critic"""


def context_feature_dim(args: Args) -> int:
    """Dimensionality of the numeric context vector (0 when mode is ``none``)."""
    if args.context_mode == "none":
        return 0
    if args.context_mode == "concat":
        return len(CONTEXT_KEYS)
    raise ValueError(f"unsupported context_mode={args.context_mode!r}")


def context_to_vector(context: Context, reference: Context) -> np.ndarray:
    """Map a context dict to a float32 vector normalized by ``reference`` scales."""
    vec = np.empty(len(CONTEXT_KEYS), dtype=np.float32)
    for i, key in enumerate(CONTEXT_KEYS):
        ref = float(reference[key])
        scale = abs(ref) if abs(ref) > 1e-8 else 1.0
        vec[i] = float(context[key]) / scale
    return vec


def build_run_name(args: Args) -> str:
    tag = base.sanitize_experiment_tag(args.experiment)
    share = "shared" if args.share_actors else "indep"
    drop = "randdrop" if args.random_drop else "nodrop"
    mode = args.context_mode
    return (
        f"{tag}__{mode}__{share}__{drop}__n{args.n_pistons}"
        f"__seed{args.seed}__{int(time.time())}"
    )


class ContextualReplayBuffer(base.MultiAgentReplayBuffer):
    """Replay buffer that also stores a per-transition context vector."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_shape: tuple[int, int, int],
        act_dim: int,
        context_dim: int,
        device: torch.device,
    ):
        super().__init__(capacity, n_agents, obs_shape, act_dim, device)
        self.context_dim = context_dim
        self.contexts = np.zeros((capacity, max(context_dim, 1)), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        reward: float,
        done: float,
        context: np.ndarray | None = None,
    ) -> None:
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context vector required when context_dim > 0")
            self.contexts[self.pos, : self.context_dim] = context
        super().add(obs, next_obs, actions, reward, done)

    def sample(self, batch_size: int):
        high = len(self)
        idx = np.random.randint(0, high, size=batch_size)
        to = lambda a: torch.as_tensor(a, device=self.device)
        obs = to(self.obs[idx]).float() / 255.0
        next_obs = to(self.next_obs[idx]).float() / 255.0
        actions = to(self.actions[idx])
        rewards = to(self.rewards[idx])
        dones = to(self.dones[idx])
        if self.context_dim > 0:
            contexts = to(self.contexts[idx, : self.context_dim])
        else:
            contexts = to(np.zeros((batch_size, 0), dtype=np.float32))
        return obs, next_obs, actions, rewards, dones, contexts

    def memory_gb(self) -> float:
        return super().memory_gb() + self.contexts.nbytes / 1e9


class ContextualActor(nn.Module):
    """Decentralized actor with optional agent-ID and context concatenation."""

    def __init__(
        self,
        args: Args,
        act_dim: int,
        action_low: float,
        action_high: float,
        n_agents: int,
        use_agent_id: bool,
        context_dim: int,
    ):
        super().__init__()
        self.use_agent_id = use_agent_id
        self.n_agents = n_agents
        self.context_mode = args.context_mode
        self.context_to_actor = args.context_to_actor and context_dim > 0
        self.context_dim = context_dim if self.context_to_actor else 0
        self.encoder = base.CNNEncoder(args.frame_stack, args.frame_size, args.feature_dim)
        in_dim = args.feature_dim
        if use_agent_id:
            in_dim += n_agents
        in_dim += self.context_dim
        self.fc1 = nn.Linear(in_dim, args.hidden_dim)
        self.fc_mu = nn.Linear(args.hidden_dim, act_dim)
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0))

    def forward(
        self,
        obs: torch.Tensor,
        agent_index: int | torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # obs: (B, C, H, W); context: (B, context_dim)
        x = self.encoder(obs)
        parts = [x]
        if self.use_agent_id:
            if agent_index is None:
                raise ValueError("shared actor requires agent_index")
            if isinstance(agent_index, int):
                idx = torch.full((obs.shape[0],), agent_index, device=obs.device, dtype=torch.long)
            else:
                idx = agent_index
            parts.append(F.one_hot(idx, self.n_agents).float())
        if self.context_to_actor:
            if context is None:
                raise ValueError("contextual actor requires context")
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"expected context dim {self.context_dim}, got {context.shape[-1]}"
                )
            parts.append(context)
        x = torch.cat(parts, dim=-1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias


class ContextualCritic(nn.Module):
    """Centralized twin-Q critic with optional context concatenation."""

    def __init__(self, args: Args, n_agents: int, act_dim: int, context_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.shared_obs = args.critic_shared_obs
        self.context_mode = args.context_mode
        self.context_to_critic = args.context_to_critic and context_dim > 0
        self.context_dim = context_dim if self.context_to_critic else 0
        self.encoder = base.CNNEncoder(args.frame_stack, args.frame_size, args.feature_dim)
        joint_dim = n_agents * (args.feature_dim + act_dim) + self.context_dim
        self.fc1 = nn.Linear(joint_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # obs: (B, N, C, H, W), actions: (B, N, act_dim), context: (B, context_dim)
        b, n = obs.shape[:2]
        if self.shared_obs:
            feats = self.encoder(obs[:, 0]).unsqueeze(1).expand(-1, n, -1).reshape(b, -1)
        else:
            feats = self.encoder(obs.reshape(b * n, *obs.shape[2:])).reshape(b, -1)
        parts = [feats, actions.reshape(b, -1)]
        if self.context_to_critic:
            if context is None:
                raise ValueError("contextual critic requires context")
            parts.append(context)
        x = torch.cat(parts, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def joint_policy_actions(
    actors: nn.ModuleList,
    obs: torch.Tensor,
    context: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stack per-agent actor outputs into a joint action ``(B, N, act_dim)``."""
    b, n = obs.shape[:2]
    actor0 = actors[0]
    if actor0.use_agent_id:
        if len(actors) != 1:
            raise ValueError("shared actor mode expects a ModuleList of length 1")
        flat_obs = obs.reshape(b * n, *obs.shape[2:])
        agent_index = torch.arange(n, device=obs.device).view(1, n).expand(b, n).reshape(b * n)
        flat_ctx = None
        if context is not None and getattr(actor0, "context_to_actor", False):
            flat_ctx = context.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
        return actor0(flat_obs, agent_index=agent_index, context=flat_ctx).reshape(b, n, -1)
    if len(actors) != n:
        raise ValueError(f"independent actors expect {n} modules, got {len(actors)}")
    outs = []
    for i, actor in enumerate(actors):
        ctx_i = context
        outs.append(actor(obs[:, i], context=ctx_i))
    return torch.stack(outs, dim=1)


@torch.no_grad()
def evaluate(
    actors: nn.ModuleList,
    args: Args,
    context: Context,
    reference_context: Context,
    agents: list[str],
    device: torch.device,
    seed: int,
    n_episodes: int,
    render_mode: str | None = None,
):
    """Roll out the deterministic contextual joint policy in ``context``."""
    env = base.make_env(args, context, render_mode=render_mode)
    ctx_vec = context_to_vector(context, reference_context)
    ctx_dim = context_feature_dim(args)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[float] = []
    episode_action_stds: list[float] = []
    episode_per_agent_var: list[float] = []
    episode_saturations: list[float] = []

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        ep_return, ep_len, success = 0.0, 0, 0.0
        ep_action_stds: list[float] = []
        ep_per_agent_vars: list[float] = []
        ep_saturations: list[float] = []
        while env.agents:
            obs = torch.as_tensor(base.stack_obs(obs_dict, agents), device=device).float() / 255.0
            if ctx_dim > 0:
                ctx_t = torch.as_tensor(ctx_vec, device=device).float().unsqueeze(0)
            else:
                ctx_t = None
            actions = joint_policy_actions(actors, obs.unsqueeze(0), context=ctx_t).squeeze(0)
            actions_np = actions.cpu().numpy()
            ep_action_stds.append(float(np.std(actions_np)))
            ep_saturations.append(
                base.action_saturation_fraction(
                    actions_np,
                    threshold=args.action_saturation_threshold,
                    action_bound=float(
                        actors[0].action_scale.item() + abs(actors[0].action_bias.item())
                    ),
                )
            )
            action_dict = {a: actions_np[i] for i, a in enumerate(agents)}
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
        episode_saturations.append(float(np.mean(ep_saturations)) if ep_saturations else 0.0)
    env.close()

    summary = summarize_episode_scores(episode_returns, reference_return=args.reference_return)
    return {
        **summary,
        "length_mean": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(episode_successes)),
        "action_std_mean": float(np.mean(episode_action_stds)),
        "action_saturation_mean": float(np.mean(episode_saturations)),
        "per_agent_return_variance_mean": float(np.mean(episode_per_agent_var)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
    }


def build_agents(args: Args, device: torch.device):
    """Construct env metadata, contextual networks, and optimizers."""
    probe_env = base.make_env(args, base.default_context(args))
    probe_env.reset(seed=args.seed)
    agents = list(probe_env.possible_agents)
    n_agents = len(agents)
    act_space = probe_env.action_space(agents[0])
    act_dim = int(np.prod(act_space.shape))
    action_low = float(act_space.low[0])
    action_high = float(act_space.high[0])
    obs_shape = (args.frame_stack, args.frame_size, args.frame_size)
    probe_env.close()

    ctx_dim = context_feature_dim(args)

    def make_actor():
        return ContextualActor(
            args,
            act_dim,
            action_low,
            action_high,
            n_agents=n_agents,
            use_agent_id=args.share_actors,
            context_dim=ctx_dim,
        ).to(device)

    if args.share_actors:
        actors = nn.ModuleList([make_actor()])
        target_actors = nn.ModuleList([make_actor()])
    else:
        actors = nn.ModuleList([make_actor() for _ in range(n_agents)])
        target_actors = nn.ModuleList([make_actor() for _ in range(n_agents)])

    qf1 = ContextualCritic(args, n_agents, act_dim, ctx_dim).to(device)
    qf2 = ContextualCritic(args, n_agents, act_dim, ctx_dim).to(device)
    qf1_target = ContextualCritic(args, n_agents, act_dim, ctx_dim).to(device)
    qf2_target = ContextualCritic(args, n_agents, act_dim, ctx_dim).to(device)
    target_actors.load_state_dict(actors.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actors.parameters(), lr=args.actor_lr)

    return {
        "agents": agents,
        "n_agents": n_agents,
        "act_dim": act_dim,
        "action_low": action_low,
        "action_high": action_high,
        "obs_shape": obs_shape,
        "context_dim": ctx_dim,
        "actors": actors,
        "target_actors": target_actors,
        "qf1": qf1,
        "qf2": qf2,
        "qf1_target": qf1_target,
        "qf2_target": qf2_target,
        "q_optimizer": q_optimizer,
        "actor_optimizer": actor_optimizer,
    }


def transfer_sweep(
    actors: nn.ModuleList,
    args: Args,
    agents: list[str],
    device: torch.device,
    reference_context: Context,
    run_dir: Path | None = None,
):
    """Evaluate the frozen contextual policy across transfer contexts."""
    contexts = base.build_transfer_contexts(args)
    context_order = list(contexts.keys())
    per_context: dict[str, dict] = {}
    scores_by_context: dict[str, list[float]] = {}
    for name, ctx in contexts.items():
        stats = evaluate(
            actors,
            args,
            ctx,
            reference_context,
            agents,
            device,
            seed=args.seed + 20_000,
            n_episodes=args.eval_episodes,
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
        "generalization_gap_mean": train_stats["return_mean"]
        - (float(np.mean(ood_means)) if ood_means else train_stats["return_mean"]),
        "generalization_gap_iqm": train_stats["return_iqm"]
        - (float(np.mean(ood_iqms)) if ood_iqms else train_stats["return_iqm"]),
        "train_return": train_stats["return_mean"],
        "ood_return_mean_legacy": float(np.mean(ood_means)) if ood_means else train_stats["return_mean"],
        "generalization_gap": train_stats["return_mean"]
        - (float(np.mean(ood_means)) if ood_means else train_stats["return_mean"]),
        "context_mode": args.context_mode,
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
                "context_mode": args.context_mode,
            },
        )
    return results


def train(args: Args) -> str:
    if args.context_mode not in ("concat", "none"):
        raise ValueError(f"unsupported --context-mode={args.context_mode!r}")

    run_name = build_run_name(args)
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.RunManifest(
        run_name=run_name,
        run_dir=run_dir,
        args=vars(args),
        environment=base.collect_environment_info(),
    )
    manifest.write()
    checkpoint_manager = (
        base.CheckpointManager(run_dir, metric_key=args.checkpoint_metric)
        if args.checkpoint_eval
        else None
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
    train_context = base.default_context(args)
    train_ctx_vec = context_to_vector(train_context, train_context)

    nets = build_agents(args, device)
    agents = nets["agents"]
    actors, target_actors = nets["actors"], nets["target_actors"]
    qf1, qf2 = nets["qf1"], nets["qf2"]
    qf1_target, qf2_target = nets["qf1_target"], nets["qf2_target"]
    q_optimizer, actor_optimizer = nets["q_optimizer"], nets["actor_optimizer"]
    action_low, action_high = nets["action_low"], nets["action_high"]
    action_scale = actors[0].action_scale
    action_bound = max(abs(action_low), abs(action_high))
    ctx_dim = nets["context_dim"]

    rb = ContextualReplayBuffer(
        args.buffer_size,
        nets["n_agents"],
        nets["obs_shape"],
        nets["act_dim"],
        ctx_dim,
        device,
    )
    buf_gb = rb.memory_gb()
    print(f"replay buffer allocation: {buf_gb:.2f} GB ({args.buffer_size} transitions)")
    print(
        f"context_mode={args.context_mode} dim={ctx_dim} "
        f"to_actor={args.context_to_actor} to_critic={args.context_to_critic}"
    )
    actor_mode = "shared+id" if args.share_actors else f"independent x{nets['n_agents']}"
    print(
        f"actors: {actor_mode} | "
        f"actor_lr={args.actor_lr} critic_lr={args.learning_rate} policy_freq={args.policy_frequency}"
    )
    if args.learning_starts >= args.buffer_size:
        raise ValueError(
            f"learning_starts ({args.learning_starts}) must be < buffer_size ({args.buffer_size}) "
            "so the random-exploration warmup is not overwritten before learning begins."
        )
    if buf_gb > 48.0:
        raise ValueError(
            f"Replay buffer would use {buf_gb:.1f} GB. Reduce --buffer-size, --n-pistons, "
            f"--frame-size, or --frame-stack."
        )

    env = base.make_env(args, train_context)
    obs_dict, _ = env.reset(seed=args.seed)
    obs = base.stack_obs(obs_dict, agents)
    ep_return, ep_len = 0.0, 0
    ep_per_agent_var_sum, ep_var_steps = 0.0, 0
    start_time = time.time()
    recent_returns: deque[float] = deque(maxlen=args.rolling_return_window)
    steps_to_threshold: int | None = None
    actor_loss = torch.tensor(0.0)
    prev_eval_returns: list[float] | None = None

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.random.uniform(
                action_low, action_high, size=(nets["n_agents"], nets["act_dim"])
            ).astype(np.float32)
            explore_std = args.exploration_noise
        else:
            explore_std = base.current_exploration_noise(args, global_step)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).float().unsqueeze(0) / 255.0
                ctx_t = (
                    torch.as_tensor(train_ctx_vec, device=device).float().unsqueeze(0)
                    if ctx_dim > 0
                    else None
                )
                actions = joint_policy_actions(actors, obs_t, context=ctx_t).squeeze(0)
                actions += torch.normal(
                    0, action_scale * explore_std, size=actions.shape, device=device
                )
                actions = actions.cpu().numpy().clip(action_low, action_high).astype(np.float32)

        action_dict = {a: actions[i] for i, a in enumerate(agents)}
        next_obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
        reward = float(np.mean(list(rewards.values())))
        rew_vals = list(rewards.values())
        ep_per_agent_var_sum += float(np.var(rew_vals))
        ep_var_steps += 1
        terminated = any(terms.values())
        truncated = any(truncs.values())
        next_obs = base.stack_obs(next_obs_dict, agents)

        ep_return += reward
        ep_len += 1

        rb.add(
            obs,
            next_obs,
            actions,
            reward,
            base.bootstrap_done(terminated, truncated),
            context=train_ctx_vec if ctx_dim > 0 else None,
        )
        obs = next_obs

        if terminated or truncated:
            ep_var_mean = ep_per_agent_var_sum / max(ep_var_steps, 1)
            writer.add_scalar("performance/team_episodic_return", ep_return, global_step)
            writer.add_scalar("performance/episodic_length", ep_len, global_step)
            writer.add_scalar("performance/success", float(terminated), global_step)
            writer.add_scalar("coordination/per_agent_return_variance", ep_var_mean, global_step)
            base.append_training_episode_record(
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
            obs = base.stack_obs(obs_dict, agents)
            ep_return, ep_len = 0.0, 0
            ep_per_agent_var_sum, ep_var_steps = 0.0, 0

        if global_step > args.learning_starts:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones, b_ctx = rb.sample(args.batch_size)

            with torch.no_grad():
                next_actions = joint_policy_actions(target_actors, b_next_obs, context=b_ctx)
                noise = (
                    (torch.randn_like(next_actions) * args.policy_noise)
                    .clamp(-args.noise_clip, args.noise_clip)
                    * action_scale
                )
                next_actions = (next_actions + noise).clamp(action_low, action_high)
                qf1_next = qf1_target(b_next_obs, next_actions, context=b_ctx)
                qf2_next = qf2_target(b_next_obs, next_actions, context=b_ctx)
                min_q_next = torch.min(qf1_next, qf2_next)
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * min_q_next

            qf1_a = qf1(b_obs, b_actions, context=b_ctx)
            qf2_a = qf2(b_obs, b_actions, context=b_ctx)
            qf_loss = F.mse_loss(qf1_a, next_q_value) + F.mse_loss(qf2_a, next_q_value)

            q_params = list(qf1.parameters()) + list(qf2.parameters())
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_grad_norm = base.gradient_norm(q_params)
            base.clip_gradients(q_params, args.max_grad_norm)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_actions = joint_policy_actions(actors, b_obs, context=b_ctx)
                actor_loss = -qf1(b_obs, actor_actions, context=b_ctx).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = base.gradient_norm(actors.parameters())
                base.clip_gradients(actors.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                base.polyak_update(actors, target_actors, args.tau)
                base.polyak_update(qf1, qf1_target, args.tau)
                base.polyak_update(qf2, qf2_target, args.tau)
            else:
                actor_grad_norm = 0.0
                actor_actions = None

            if global_step % 100 == 0:
                base.log_timing_metrics(writer, global_step, start_time, args.total_timesteps)
                writer.add_scalar("losses/qf1_values", qf1_a.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("exploration/noise_std", explore_std, global_step)
                sat_actions = (
                    actor_actions
                    if actor_actions is not None
                    else joint_policy_actions(actors, b_obs, context=b_ctx)
                )
                writer.add_scalar(
                    "stability/action_saturation",
                    base.action_saturation_fraction(
                        sat_actions,
                        threshold=args.action_saturation_threshold,
                        action_bound=action_bound,
                    ),
                    global_step,
                )
                if args.log_gradient_norms:
                    writer.add_scalar("stability/q_grad_norm", q_grad_norm, global_step)
                    writer.add_scalar("stability/actor_grad_norm", actor_grad_norm, global_step)
                elapsed = time.time() - start_time
                print(f"SPS: {int(global_step / elapsed)} | wallclock: {elapsed / 3600:.2f} h")

        if (
            args.eval_frequency > 0
            and global_step > args.learning_starts
            and global_step % args.eval_frequency == 0
        ):
            eval_start = time.time()
            stats = evaluate(
                actors,
                args,
                train_context,
                train_context,
                agents,
                device,
                seed=args.seed + 10_000,
                n_episodes=args.eval_episodes,
            )
            eval_wallclock = time.time() - eval_start
            total_wallclock = time.time() - start_time
            stats["eval_wallclock_seconds"] = eval_wallclock
            stats["total_wallclock_seconds"] = total_wallclock
            base.log_eval_metrics(writer, "eval_id", stats, global_step)
            writer.add_scalar("time/eval_wallclock_seconds", eval_wallclock, global_step)
            writer.add_scalar("time/total_wallclock_seconds", total_wallclock, global_step)
            if prev_eval_returns is not None and stats["episode_returns"] == prev_eval_returns:
                writer.add_scalar("stability/eval_returns_unchanged", 1.0, global_step)
                print(
                    f"[warn @ {global_step}] eval episode returns identical to previous eval "
                    "(deterministic policy likely frozen / saturated)"
                )
            else:
                writer.add_scalar("stability/eval_returns_unchanged", 0.0, global_step)
            prev_eval_returns = list(stats["episode_returns"])
            base.append_eval_record(run_dir, global_step, stats, total_wallclock)
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
                extra={
                    "checkpoint_metric": args.checkpoint_metric,
                    "experiment": args.experiment,
                    "context_mode": args.context_mode,
                },
            )
            print(
                f"[eval @ {global_step}] return={stats['return_mean']:.2f} "
                f"success={stats['success_rate']:.2f} "
                f"sat={stats['action_saturation_mean']:.2f} "
                f"eval_time={eval_wallclock:.1f}s total_time={total_wallclock / 3600:.2f}h"
            )
            if checkpoint_manager is not None:
                saved = checkpoint_manager.save(
                    base.build_checkpoint_payload(nets, args),
                    global_step,
                    stats,
                )
                print(
                    f"[checkpoint] latest={saved['latest']}"
                    + (f" best={saved['best']}" if "best" in saved else "")
                )
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

    restored_best_step = base.maybe_restore_best_checkpoint(checkpoint_manager, nets, device)
    actors = nets["actors"]

    if args.transfer_eval:
        transfer_start = time.time()
        results = transfer_sweep(
            actors, args, agents, device, train_context, run_dir=run_dir
        )
        results["transfer_wallclock_seconds"] = time.time() - transfer_start
        results["total_wallclock_seconds"] = time.time() - start_time
        if restored_best_step is not None:
            results["checkpoint_step"] = restored_best_step
        with open(run_dir / "transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)
        base.log_transfer_results(writer, results, args.total_timesteps)
        print("transfer sweep:", json.dumps(results["contexts"], indent=2))
        print(f"generalization gap: {results['generalization_gap']:.3f}")

    if args.save_model:
        model_path = run_dir / f"{args.exp_name}_final.pt"
        torch.save(base.build_checkpoint_payload(nets, args), model_path)
        print(f"model saved to {model_path}")

    manifest.global_step = args.total_timesteps
    manifest.wallclock_seconds = time.time() - start_time
    manifest.finished = True
    manifest.write()
    print(f"run finished in {manifest.wallclock_seconds / 3600:.2f} h -> {run_dir}")

    writer.close()
    return run_name


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
