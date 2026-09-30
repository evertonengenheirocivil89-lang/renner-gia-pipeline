# renner-gia-pipeline

Financial analysis pipeline for Lojas Renner S.A. - ETL, forecasting, risk, ESG, and RL modules.

## RL optimization update

This repository now includes a reinforcement learning layer focused on stable policy learning for financial decisions.

### Optimized defaults

- learning rate: 3e-4
- discount factor: 0.99
- epsilon start: 1.0
- epsilon minimum: 0.05
- epsilon decay: 0.995
- batch size: 128
- replay buffer: 100000
- target update interval: 1000

### Structure added

- `src/rl/__init__.py`
- `src/rl/hyperparameters.py`
- `src/rl/reinforcement_learning.py`
- `src/rl/training.py`

### Example

```python
from src.rl.hyperparameters import RLHyperparameters
from src.rl.reinforcement_learning import (
    AdaptiveQLearningAgent,
    FinancialMarketEnvironment,
    optimize_hyperparameters,
)

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
```

This configuration improves exploration in early stages and stabilizes learning in later stages, which is important for noisy financial signals and risk-aware decisions.
