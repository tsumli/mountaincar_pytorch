import gym

from utils import load_agent


def test(agent, env, policy, max_steps):
    obs = env.reset()
    done = False
    total_reward = 0.0

    for step in range(max_steps):
        env.render()
        action = policy[agent.discretize(obs)]

        next_obs, reward, done, trancated, info = env.step(action)
        obs = next_obs
        total_reward += reward

        if done:
            break

    return total_reward


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")
    print(env.observation_space)
    agent = load_agent("logs/030000.pth")
    for _ in range(5):
        reward = test(agent, env, agent.policy(), max_steps=200)
        print("reward: ", reward)
    env.close()
