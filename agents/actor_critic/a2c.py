import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..base import BaseAgent
from ..eps_greedy import EpsGreedy
from .net import ActorCriticNNModule
from ..utils import ReplayMemory, Transition


class A2CAgent(BaseAgent):
    def __init__(
        self,
        env,
        batch_size: int,
        lr: float,
        gamma: float,
        device: str,
        **kwargs
    ):
        super().__init__()
        space_dim = len(env.observation_space.high)
        n_actions = env.action_space.n
        policy_net = ActorCriticNNModule(space_dim, n_actions, device=device).to(device)
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.scheduler = None

        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device

        # Loss
        self.criterion = nn.SmoothL1Loss()

        self.steps_done = 0
        self.learns_done = 0
        self.before_episode = 0

        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.probs = []

    @torch.no_grad()
    def select_action(
        self, state
    ):
        self.policy_net.eval()
        if isinstance(state, tuple):
            state = state[0]
        state = torch.as_tensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action, _ = self.policy_net(state)
        action = torch.argmax(action, dim=1)
        return action

    def _tensor_state(self, state) -> torch.Tensor:
        if isinstance(state, tuple):
            state = state[0]
        state = torch.as_tensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state

    def learn(self, env, state, episode, step):
        self.policy_net.train()
        state = self._tensor_state(state)

        prob, value = self.policy_net(state)
        action = prob.multinomial(1).squeeze(0).item()
        next_state, reward, done, _, _ = env.step(
            action
        )
        self.states.append(next_state)
        self.rewards.append(reward)
        self.values.append(value)
        self.actions.append(action)
        self.probs.append(prob)

        _, v = local_policy(o)
        R += v.data.squeeze()[0]

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        if self.before_episode != episode:
            if self.scheduler is not None:
                self.scheduler.step()
        self.before_episode = episode

        self.learns_done += 1
