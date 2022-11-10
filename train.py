from pathlib import Path

import gym
import torch
from tqdm import tqdm

from utils import load_config, make_agent, save_agent


def train(
    env,
    agent,
    max_episodes: int,
    max_steps: int,
    agent_name: str,
    log_dir=Path("logs"),
    save_frequency=5000,
):
    best_reward = -float("inf")
    log_dir = Path(log_dir)

    with tqdm(range(1, int(max_episodes) + 1)) as pbar:
        for episode in range(1, int(max_episodes) + 1):
            state = env.reset()
            done = False
            total_reward = 0.0

            for step in range(1, int(max_steps) + 1):
                action = agent.select_action(state)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().detach().numpy()
                next_state, reward, done, truncated, info = env.step(action.item(0))

                agent.learn(state, action, reward, next_state, episode, step)
                state = next_state
                total_reward += reward

                if done:
                    break

            # ベスト報酬の更新
            if total_reward > best_reward:
                best_reward = total_reward
                save_agent(agent, log_dir / agent_name / "best_reward.pth")

            if episode % save_frequency == 0:
                save_agent(agent, log_dir / agent_name / f"{str(episode).zfill(5)}.pth")

            pbar.set_postfix(
                {
                    "reward": total_reward,
                    "best_reward": best_reward,
                }
            )
            pbar.update(1)
        save_agent(agent, log_dir / agent_name / f"{str(episode).zfill(5)}.pth")


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    config = load_config("config.yaml")
    agent_name = config.train.agent_name
    cls_agent = make_agent(agent_name)
    agent = cls_agent(env, **config[agent_name])
    train(env, agent, **config.train)
    env.close()
