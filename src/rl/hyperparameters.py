from dataclasses import dataclass
from typing import Tuple


@dataclass
class RLHyperparameters:
    """Hyperparameters tuned for financial reinforcement learning.

    These values balance exploration and stability in non-stationary market-like
    environments, helping the agent converge without excessive instability.
    """

    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 128
    replay_buffer_size: int = 100_000
    target_update_interval: int = 1_000
    warmup_steps: int = 1_000
    max_steps_per_episode: int = 200
    n_episodes: int = 500
    reward_scaling: float = 1.0
    gradient_clip_norm: float = 5.0
    l2_regularization: float = 1e-4
    action_space: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    state_bins: Tuple[int, ...] = (12, 12, 12, 12)


DEFAULT_RL_CONFIG = RLHyperparameters()
