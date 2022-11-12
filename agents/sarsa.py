from typing import Optional

import numpy as np

from .base import BaseAgent
from .eps_greedy import EpsGreedy


class SARSAAgent(BaseAgent):
    def __init__(
        self,
        env,
        gamma: float,
        alpha: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        num_bins: int = 30,
        **kwargs
    ):
        super().__init__()
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.action_shape = env.action_space.n

        self.bin_width = (self.obs_high - self.obs_low) / num_bins
        self.Q = np.zeros((num_bins + 1, num_bins + 1, self.action_shape))
        self.eps = 1.0
        self.gamma = gamma
        self.alpha = alpha
        self.eps_greedy = EpsGreedy(
            eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay
        )

    def discretize(self, state):
        if isinstance(state, tuple):
            state = state[0]
        return tuple(((state - self.obs_low) / self.bin_width).astype(int))

    def select_action(
        self,
        state,
        eps_threshold: Optional[float] = None,
        eps_update: bool = True,
        discretize: bool = True,
    ):
        if discretize:
            state = self.discretize(state)
        if eps_threshold is None:
            eps_threshold = self.eps_greedy.get_threshold()
        if eps_update:
            self.eps_greedy.update_step()

        if np.random.random() > eps_threshold:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, state, action, reward, next_state, episode, step):
        state = self.discretize(state)
        next_state = self.discretize(next_state)
        next_action = self.select_action(next_state, eps_update=False, discretize=False)

        td_target = reward + self.gamma * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
