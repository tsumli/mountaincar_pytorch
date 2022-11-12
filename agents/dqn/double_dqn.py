from typing import Optional
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from ..base import BaseAgent
from ..eps_greedy import EpsGreedy
from .net import NNModule
from .utils import ReplayMemory, Transition


class DoubleDQNAgent(BaseAgent):
    def __init__(
        self,
        env,
        batch_size: int,
        lr: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        target_update: int,
        device: str,
        **kwargs
    ):
        # Get number of actions from gym action space
        space_dim = len(env.observation_space.high)
        n_actions = env.action_space.n
        policy_net = NNModule(space_dim, n_actions, device=device).to(device)
        target_net = NNModule(space_dim, n_actions, device=device).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        self.policy_net = policy_net
        self.target_net = target_net

        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.scheduler = None
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=[300, 800, 1300], gamma=0.8
        # )

        self.memory = ReplayMemory(10000)

        self.batch_size = batch_size
        self.eps_greedy = EpsGreedy(
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
        )
        self.target_update = target_update
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device

        # Loss
        self.criterion = nn.SmoothL1Loss()

        self.steps_done = 0
        self.learns_done = 0
        self.before_episode = 0

    def select_action(
        self, state, eps_threshold: Optional[float] = None, eps_update: bool = True
    ):
        sample = random.random()

        if eps_threshold is None:
            eps_threshold = self.eps_greedy.get_threshold()
        if eps_update:
            self.eps_greedy.update_step()

        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval()
                if isinstance(state, tuple):
                    state = state[0]
                state = torch.as_tensor(state)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def learn(self, state, action, reward, next_state, episode, step):
        self.policy_net.train()
        if isinstance(state, tuple):
            state = state[0]

        self.memory.push(
            torch.as_tensor(state),
            torch.as_tensor(action),
            torch.as_tensor(reward).view(-1),
            torch.as_tensor(next_state),
        )

        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

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

        if self.learns_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learns_done += 1
