"""Per-agent context features for the Pistonball context-kind ablation.

Physics context is a global episode vector. Agent context is a *per-step*
oracle of local geometry (relative ball x, piston y) that a visual policy
could in principle infer from pixels. Concatenating it tests whether explicit
local coordinates help coordination even when physics OOD is uninformative.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.context_sampling import get_raw_pistonball_env

AGENT_FEATURE_KEYS: tuple[str, ...] = ("rel_ball_x", "piston_y")
AGENT_FEATURE_DIM: int = len(AGENT_FEATURE_KEYS)


def pistonball_agent_features(env: Any, agents: list[str]) -> np.ndarray:
    """Return ``(N, 2)`` features: ball-x relative to piston, piston y. Both in [~-1, 1]."""
    raw = get_raw_pistonball_env(env)
    ball_x = float(raw.ball.position[0])
    width = float(getattr(raw, "screen_width", 1.0)) or 1.0
    height = float(getattr(raw, "screen_height", 1.0)) or 1.0
    mapping = getattr(raw, "agent_name_mapping", None) or {
        name: i for i, name in enumerate(agents)
    }
    feats = np.zeros((len(agents), AGENT_FEATURE_DIM), dtype=np.float32)
    for i, name in enumerate(agents):
        idx = int(mapping[name]) if name in mapping else i
        body = raw.pistonList[idx]
        px = float(body.position[0])
        py = float(body.position[1])
        feats[i, 0] = (ball_x - px) / width
        feats[i, 1] = py / height
    return feats
