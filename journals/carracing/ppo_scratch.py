import argparse
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStackObservation
from typing import Callable, cast

from utils.paths import RUNS_DIR


# ----------------------
# Optimized preprocessing
# ----------------------
class ResizeObsWrapper(gym.ObservationWrapper):
    """Resize CarRacing frames efficiently."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Only works with Box spaces"
        self.width = width
        self.height = height
        self.observation_space = Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs)
        return cv2.resize(
            obs, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )


class GrayscaleObsWrapper(gym.ObservationWrapper):
    """Convert to grayscale to reduce input size."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = Box(
            low=0, high=255, shape=(*old_space.shape[:2], 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs[:, :, np.newaxis]


class NormalizeObsWrapper(gym.ObservationWrapper):
    """Normalize pixel values to [0,1] with faster conversion."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class ChannelFirstWrapper(gym.ObservationWrapper):
    """Convert observation to PyTorch format (C * num_stack, H, W)"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        old_shape = env.observation_space.shape  # e.g., (4, 84, 84, 1)
        assert len(old_shape) == 4, f"Expected (num_stack, H, W, C), got {old_shape}"
        num_stack, h, w, c = old_shape
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(c * num_stack, h, w), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs shape: (num_stack, H, W, C)
        num_stack, h, w, c = obs.shape
        obs = obs.transpose(1, 2, 0, 3)  # (H, W, num_stack, C)
        obs = obs.reshape(h, w, -1)  # (H, W, C*num_stack)
        return obs.transpose(2, 0, 1)  # (C*num_stack, H, W)


# ----------------------
# Improved Agent Architecture
# ----------------------
class CNNAgent(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        c, h, w = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute feature size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out_size = self.cnn(dummy).shape[1]

        self.policy_net = nn.Sequential(
            nn.Linear(out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.zero_()

    def get_value(self, x):
        features = self.value_net(self.cnn(x))
        return self.critic(features)

    def get_action_and_value(self, x):
        cnn_features = self.cnn(x)

        # Policy head
        policy_features = self.policy_net(cnn_features)
        mean = self.actor_mean(policy_features)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        # Value head
        value_features = self.value_net(cnn_features)
        value = self.critic(value_features)

        return dist, value


# ----------------------
# Training Loop Optimizations
# ----------------------
def make_raw_env(env_id: str, render_mode: str | None = None):
    return gym.make(env_id, render_mode=render_mode, continuous=True)


def make_env(env_id: str, resize_shape=(84, 84)) -> Callable[[], gym.Env]:
    def thunk():
        env = make_raw_env(env_id)
        env = ResizeObsWrapper(env, width=resize_shape[0], height=resize_shape[1])
        env = GrayscaleObsWrapper(env)
        env = NormalizeObsWrapper(env)
        env = FrameStackObservation(env, 4)
        env = ChannelFirstWrapper(env)
        env = RecordEpisodeStatistics(env)
        return env

    return thunk


def train(args):
    writer = SummaryWriter(
        f"{RUNS_DIR}/carracing/ppo_optimized_{args.env_id.lower()}_{int(time.time())}"
    )

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Create environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, resize_shape=(84, 84)) for _ in range(args.num_envs)]
    )

    obs_space = cast(Box, envs.single_observation_space)
    act_space = cast(Box, envs.single_action_space)

    print(f"Observation space shape: {obs_space.shape}")  # Should be (4, 84, 84)

    # Use the shape directly — it's already (C, H, W) after ChannelFirstWrapper
    obs_shape_for_cnn = obs_space.shape
    act_dim = act_space.shape[0]

    agent = CNNAgent(obs_shape_for_cnn, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Pre-allocate buffers
    obs = torch.zeros(
        (args.num_steps_per_rollout, args.num_envs) + obs_shape_for_cnn, device=device
    )
    actions = torch.zeros(
        (args.num_steps_per_rollout, args.num_envs, act_dim), device=device
    )
    logprobs = torch.zeros((args.num_steps_per_rollout, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps_per_rollout, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps_per_rollout, args.num_envs), device=device)
    values = torch.zeros((args.num_steps_per_rollout, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    for iteration in range(args.num_iterations):
        for step in range(args.num_steps_per_rollout):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                dist, value = agent.get_action_and_value(next_obs)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(axis=-1)

            values[step] = value.view(-1)
            actions[step] = action
            logprobs[step] = logprob

            # Step environments
            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminated, truncated)

            # Log episode info
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        print(
                            f"global_step={global_step}, episodic_return={episodic_return}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", episodic_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            global_step += args.num_envs

        # Compute advantages with GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze() * (1 - next_done)
            advantages = torch.zeros_like(rewards, device=device)
            gae = 0
            for t in reversed(range(args.num_steps_per_rollout)):
                mask = 1 - dones[t]
                delta = rewards[t] + args.gamma * next_value * mask - values[t]
                gae = delta + args.gamma * args.gae_lambda * mask * gae
                advantages[t] = gae
                next_value = values[t]
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape_for_cnn)
        b_actions = actions.reshape((-1, act_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # PPO optimization
        for epoch in range(args.epochs):
            indices = torch.randperm(b_obs.size(0), device=device)

            for start in range(0, len(indices), args.minibatch_size):
                end = start + args.minibatch_size
                mb_indices = indices[start:end]

                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_logprobs = b_logprobs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]

                # Get new policy
                dist, mb_values = agent.get_action_and_value(mb_obs)
                mb_new_logprobs = dist.log_prob(mb_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                # Policy loss
                ratio = (mb_new_logprobs - mb_old_logprobs).exp()
                # Note: Re-normalizing per minibatch is optional; you can remove if desired
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                mb_values = mb_values.view(-1)
                v_loss_unclipped = (mb_values - mb_returns) ** 2
                v_clipped = mb_returns + torch.clamp(
                    mb_values - mb_returns, -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Total loss
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)

        sps = int(global_step / (time.time() - start_time))
        print(f"Iteration {iteration + 1}/{args.num_iterations}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

        # Save checkpoint periodically
        if (iteration + 1) % 50 == 0:
            torch.save(agent.state_dict(), f"ppo_{args.env_id.lower()}_checkpoint.pt")

    # Final save
    torch.save(agent.state_dict(), f"ppo_{args.env_id.lower()}_final.pt")
    envs.close()
    writer.close()


# ----------------------
# Evaluation
# ----------------------
def evaluate(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env = make_raw_env(args.env_id, render_mode="human")
    env = ResizeObsWrapper(env, width=84, height=84)
    env = GrayscaleObsWrapper(env)
    env = NormalizeObsWrapper(env)
    env = FrameStackObservation(env, 4)
    env = ChannelFirstWrapper(env)  # Critical: match training preprocessing

    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    policy = CNNAgent(obs_shape, act_dim).to(device)
    policy.load_state_dict(
        torch.load(f"ppo_{args.env_id.lower()}_final.pt", map_location=device)
    )
    policy.eval()

    obs, _ = env.reset()
    obs = torch.Tensor(obs).to(device)
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            dist, _ = policy.get_action_and_value(obs.unsqueeze(0))
            action = (
                dist.mean.squeeze(0).cpu().numpy()
            )  # Use mean for deterministic eval

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = torch.Tensor(obs).to(device)

    print(f"Evaluation completed. Total reward: {total_reward}")
    env.close()


class Args(argparse.Namespace):
    seed: int
    torch_deterministic: bool
    total_timesteps: int
    num_envs: int
    num_steps_per_rollout: int
    gamma: float
    gae_lambda: float
    num_minibatches: int
    epochs: int
    learning_rate: float
    clip_coef: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float
    env_id: str
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--torch-deterministic", type=bool, default=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps-per-rollout", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")

    args = parser.parse_args(namespace=Args())
    args.batch_size = args.num_envs * args.num_steps_per_rollout
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    train(args)
    evaluate(args)
