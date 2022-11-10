import numpy as np


class QAgent(object):
    def __init__(
        self,
        env,
        gamma: float,
        alpha: float,
        epsilon_min: float,
        epsilon_decay: float,
        num_bins: int = 30,
        **kwargs
    ):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.action_shape = env.action_space.n

        self.bin_width = (self.obs_high - self.obs_low) / num_bins
        self.Q = np.zeros((num_bins + 1, num_bins + 1, self.action_shape))
        self.epsilon = 1.0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def discretize(self, state):
        if isinstance(state, tuple):
            state = state[0]
        return tuple(((state - self.obs_low) / self.bin_width).astype(int))

    def select_action(self, obs):
        state = self.discretize(obs)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        if np.random.random() > self.epsilon_min:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, state, action, reward, next_state, episode, step):
        state = self.discretize(state)
        next_state = self.discretize(next_state)

        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def policy(self):
        return np.argmax(self.Q, axis=2)
