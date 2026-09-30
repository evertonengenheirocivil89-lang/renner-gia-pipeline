import numpy as np

from src.rl.hyperparameters import RLHyperparameters
from src.rl.reinforcement_learning import AdaptiveQLearningAgent, FinancialMarketEnvironment, optimize_hyperparameters


def run_training() -> dict:
    config = RLHyperparameters(
        learning_rate=3e-4,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )
    env = FinancialMarketEnvironment(feature_dim=4)
    best_config = optimize_hyperparameters(env, episodes_per_config=20)
    agent = AdaptiveQLearningAgent(action_space=config.action_space, config=best_config)
    score = agent.evaluate_policy(env, episodes=25)
    return {
        "best_hyperparameters": {
            "learning_rate": best_config.learning_rate,
            "discount_factor": best_config.discount_factor,
            "epsilon_start": best_config.epsilon_start,
            "epsilon_min": best_config.epsilon_min,
            "epsilon_decay": best_config.epsilon_decay,
        },
        "average_reward": round(float(score), 4),
    }


if __name__ == "__main__":
    print(run_training())
