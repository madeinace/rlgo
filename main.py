import gymnasium as gym
import logging

logger = logging.getLogger()


env = gym.make("CliffWalking-v0", render_mode="human")


def main():
    print("Hello from rl-cliff-walking!")
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            break
    env.close()


if __name__ == "__main__":
    main()
