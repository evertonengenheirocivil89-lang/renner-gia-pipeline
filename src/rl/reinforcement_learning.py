import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.rl.hyperparameters import DEFAULT_RL_CONFIG, RLHyperparameters


class FinancialMarketEnvironment:
    """Synthetic market environment for testing RL policies."""

    def __init__(self, feature_dim: int = 4, volatility: float = 0.15):
        self.feature_dim = feature_dim
        self.volatility = volatility
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self._generate_state()

    def _generate_state(self) -> np.ndarray:
        drift = 0.02 + 0.05 * math.sin(self.current_step / 4)
        trend = 0.01 * self.current_step
        volatility_component = self.volatility * (0.5 + 0.5 * math.sin(self.current_step / 3))
        base = np.linspace(-1.0, 1.0, self.feature_dim, dtype=float)
        signal = base + drift + trend + volatility_component
        return np.clip(signal, -3.0, 3.0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        next_state = self._generate_state()
        market_signal = float(np.mean(next_state))
        reward = 0.0

        reward += market_signal * (0.5 + abs(action) * 0.1)
        reward -= abs(action) * 0.08
        reward -= abs(market_signal) * 0.06
        reward += 0.1 if action > 0 and market_signal > 0 else 0.0
        reward -= 0.1 if action < 0 and market_signal < 0 else 0.0

        done = self.current_step >= 200
        info = {"market_signal": market_signal, "step": self.current_step}
        return next_state, reward, done, info


class AdaptiveQLearningAgent:
    """Tabular Q-learning agent tuned for financial signals."""

    def __init__(
        self,
        action_space: Sequence[int],
        config: RLHyperparameters | None = None,
        state_bins: Sequence[int] = (12, 12, 12, 12),
    ):
        self.config = config or DEFAULT_RL_CONFIG
        self.action_space = list(action_space)
        self.state_bins = tuple(int(x) for x in state_bins)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.epsilon = self.config.epsilon_start
        self.training_steps = 0

    def _discretize_state(self, state: Sequence[float]) -> Tuple[int, ...]:
        if len(state) != len(self.state_bins):
            raise ValueError(
                f"Expected {len(self.state_bins)} state dimensions, got {len(state)}"
            )

        discretized: List[int] = []
        for value, bins in zip(state, self.state_bins):
            bounded = max(-3.0, min(3.0, float(value)))
            idx = int((bounded + 3.0) / 6.0 * bins)
            idx = max(0, min(bins - 1, idx))
            discretized.append(idx)
        return tuple(discretized)

    def _ensure_q_values(self, key: Tuple[int, ...]) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.action_space), dtype=float)
        return self.q_table[key]

    def act(self, state: Sequence[float]) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        key = self._discretize_state(state)
        q_values = self._ensure_q_values(key)
        action_index = int(np.argmax(q_values))
        return self.action_space[action_index]

    def learn(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> float:
        current_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state)

        action_index = self.action_space.index(action)
        current_q = self._ensure_q_values(current_key)
        next_q = self._ensure_q_values(next_key)

        target = reward
        if not done:
            target += self.config.discount_factor * float(np.max(next_q))

        current_q[action_index] += self.config.learning_rate * (
            target - current_q[action_index]
        )

        current_q *= 1.0 - self.config.l2_regularization

        self.training_steps += 1
        if self.training_steps >= self.config.warmup_steps:
            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon * self.config.epsilon_decay,
            )

        return float(target)

    def evaluate_policy(self, env: FinancialMarketEnvironment, episodes: int = 10) -> float:
        total_reward = 0.0
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
        return total_reward / max(1, episodes)


def optimize_hyperparameters(
    env: FinancialMarketEnvironment,
    candidate_configs: Iterable[RLHyperparameters] | None = None,
    episodes_per_config: int = 20,
) -> RLHyperparameters:
    """Performs a lightweight search over RL hyperparameters."""

    configs = list(candidate_configs) if candidate_configs else [
        RLHyperparameters(learning_rate=1e-4, discount_factor=0.95, epsilon_decay=0.995),
        RLHyperparameters(learning_rate=3e-4, discount_factor=0.97, epsilon_decay=0.995),
        RLHyperparameters(learning_rate=3e-4, discount_factor=0.99, epsilon_decay=0.995),
        RLHyperparameters(learning_rate=5e-4, discount_factor=0.99, epsilon_decay=0.997),
        RLHyperparameters(learning_rate=3e-4, discount_factor=0.99, epsilon_decay=0.999),
    ]

    best_config = configs[0]
    best_score = -float("inf")

    for config in configs:
        agent = AdaptiveQLearningAgent(action_space=config.action_space, config=config)
        rewards = []
        for _ in range(episodes_per_config):
            state = env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
            rewards.append(episode_reward)

        mean_reward = float(np.mean(rewards))
        if mean_reward > best_score:
            best_score = mean_reward
            best_config = config

    return best_config


if __name__ == "__main__":
    env = FinancialMarketEnvironment(feature_dim=4)
    config = optimize_hyperparameters(env, episodes_per_config=20)
    agent = AdaptiveQLearningAgent(action_space=config.action_space, config=config)
    score = agent.evaluate_policy(env, episodes=25)
    print({
        "best_hyperparameters": {
            "learning_rate": config.learning_rate,
            "discount_factor": config.discount_factor,
            "epsilon_start": config.epsilon_start,
            "epsilon_min": config.epsilon_min,
            "epsilon_decay": config.epsilon_decay,
        },
        "average_reward": round(float(score), 4),
    })
