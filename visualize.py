import gym
import torch

from utils import load_agent


def visualize(agent, env, max_steps):
    state = env.reset()
    done = False
    total_reward = 0.0

    for _ in range(max_steps):
        env.render()
        action = agent.select_action(state)
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()

        next_state, reward, done, trancated, info = env.step(action.item(0))
        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    env = gym.make("MountainCar-v0", render_mode="human")
    agent = load_agent(args.path)
    for _ in range(5):
        reward = visualize(agent, env, max_steps=200)
        print("reward: ", reward)
    env.close()
