import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordEpisodeStatistics
from typing import Callable, cast

from utils.paths import RUNS_DIR


# ----------------------
# GPU Preprocessing (NO OpenCV, NO float64)
# ----------------------
def preprocess_obs(obs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Input:  (B, H, W, C) uint8 [0-255] on CPU
    Output: (B, 84, 84) float32 [0-1] on GPU
    """
    # Convert to float32 early, normalize, move to GPU
    obs = obs.to(device=device, dtype=torch.float32) / 255.0  # (B, 96, 96, 3)

    # Grayscale conversion (luminance)
    gray = 0.299 * obs[:, :, :, 0] + 0.587 * obs[:, :, :, 1] + 0.114 * obs[:, :, :, 2]
    gray = gray.unsqueeze(1)  # (B, 1, 96, 96)

    # Resize to 84x84 using bilinear interpolation (GPU-accelerated)
    gray = torch.nn.functional.interpolate(
        gray, size=(84, 84), mode="bilinear", align_corners=False
    )  # (B, 1, 84, 84)

    return gray.squeeze(1)  # (B, 84, 84)


# ----------------------
# Agent
# ----------------------
class CNNAgent(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        c, h, w = obs_shape  # e.g., (4, 84, 84)

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
        policy_features = self.policy_net(cnn_features)
        mean = self.actor_mean(policy_features)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        value_features = self.value_net(cnn_features)
        value = self.critic(value_features)
        return dist, value


# ----------------------
# Environment Factory
# ----------------------
def make_raw_env(env_id: str, render_mode: str | None = None):
    return gym.make(env_id, render_mode=render_mode, continuous=True)


def make_env(env_id: str) -> Callable[[], gym.Env]:
    def thunk():
        env = make_raw_env(env_id)
        env = RecordEpisodeStatistics(env)
        return env

    return thunk


# ----------------------
# Training
# ----------------------
def train(args):
    writer = SummaryWriter(
        f"{RUNS_DIR}/carracing/ppo_mps_{args.env_id.lower()}_{int(time.time())}"
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic and device.type == "cuda":
        torch.backends.cudnn.deterministic = True

    # Use AsyncVectorEnv for better overlap
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)]
    )

    act_space = cast(Box, envs.single_action_space)
    act_dim = act_space.shape[0]

    agent = CNNAgent((4, 84, 84), act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Frame stack buffer (on GPU)
    frame_stack = torch.zeros(
        (args.num_envs, 4, 84, 84), dtype=torch.float32, device=device
    )

    # Rollout buffers
    buffer_size = args.num_steps_per_rollout
    obs_buf = torch.zeros((buffer_size, args.num_envs, 4, 84, 84), device=device)
    actions_buf = torch.zeros((buffer_size, args.num_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((buffer_size, args.num_envs), device=device)
    rewards_buf = torch.zeros((buffer_size, args.num_envs), device=device)
    dones_buf = torch.zeros((buffer_size, args.num_envs), device=device)
    values_buf = torch.zeros((buffer_size, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()

    # Reset environments
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(next_obs_np.astype(np.uint8))  # Keep uint8 on CPU
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    # Initialize frame stack with first observation
    preprocessed = preprocess_obs(next_obs, device)
    frame_stack[:, -1] = preprocessed

    for iteration in range(args.num_iterations):
        frame_stack.zero_()

        for step in range(buffer_size):
            # Shift frames: keep last 3, add new one at end
            frame_stack[:, :3] = frame_stack[:, 1:].clone()
            preprocessed = preprocess_obs(next_obs, device)
            frame_stack[:, 3] = preprocessed

            obs_buf[step] = frame_stack.clone()
            dones_buf[step] = next_done

            with torch.no_grad():
                dist, value = agent.get_action_and_value(frame_stack)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(dim=-1)

            values_buf[step] = value.squeeze()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step environments
            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            next_obs = torch.from_numpy(next_obs_np.astype(np.uint8))

            next_done_np = np.logical_or(terminated, truncated)
            next_done = torch.as_tensor(
                next_done_np, dtype=torch.float32, device=device
            )
            rewards_buf[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=device
            )

            global_step += args.num_envs

            # Log episode stats
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_return = info["episode"]["r"]
                        episode_length = info["episode"]["l"]
                        print(
                            f"global_step={global_step}, episodic_return={episode_return:.1f}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", episode_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", episode_length, global_step
                        )

        # Bootstrap value
        with torch.no_grad():
            preprocessed = preprocess_obs(next_obs, device)
            frame_stack[:, :3] = frame_stack[:, 1:]
            frame_stack[:, 3] = preprocessed
            next_value = agent.get_value(frame_stack).squeeze() * (1 - next_done)
            advantages = torch.zeros_like(rewards_buf, device=device)
            gae = 0
            for t in reversed(range(buffer_size)):
                mask = 1 - dones_buf[t]
                delta = rewards_buf[t] + args.gamma * next_value * mask - values_buf[t]
                gae = delta + args.gamma * args.gae_lambda * mask * gae
                advantages[t] = gae
                next_value = values_buf[t]
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape((-1, 4, 84, 84))
        b_actions = actions_buf.reshape((-1, act_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # PPO update
        for epoch in range(args.epochs):
            indices = torch.randperm(b_obs.size(0), device=device)
            for start in range(0, len(indices), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = indices[start:end]

                dist, mb_values = agent.get_action_and_value(b_obs[mb_inds])
                mb_logprobs = dist.log_prob(b_actions[mb_inds]).sum(dim=-1)
                mb_entropy = dist.entropy().sum(dim=-1).mean()
                ratio = (mb_logprobs - b_logprobs[mb_inds]).exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                mb_values = mb_values.view(-1)
                v_loss_unclipped = (mb_values - b_returns[mb_inds]) ** 2
                v_clipped = b_returns[mb_inds] + torch.clamp(
                    mb_values - b_returns[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * mb_entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        y_pred, y_true = (
            mb_values.detach().cpu().numpy(),
            b_returns[mb_inds].cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"Iteration {iteration + 1}/{args.num_iterations} | SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", mb_entropy.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if (iteration + 1) % 50 == 0:
            torch.save(agent.state_dict(), f"ppo_{args.env_id.lower()}_checkpoint.pt")

    torch.save(agent.state_dict(), f"ppo_{args.env_id.lower()}_final.pt")
    envs.close()
    writer.close()


# ----------------------
# Evaluation
# ----------------------
def evaluate(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    env = make_raw_env(args.env_id, render_mode="human")
    obs, _ = env.reset()
    obs = torch.from_numpy(obs.astype(np.uint8)).unsqueeze(0)  # (1, H, W, C)

    agent = CNNAgent((4, 84, 84), env.action_space.shape[0]).to(device)
    agent.load_state_dict(
        torch.load(f"ppo_{args.env_id.lower()}_final.pt", map_location=device)
    )
    agent.eval()

    frame_stack = torch.zeros(1, 4, 84, 84, device=device)
    total_reward = 0
    done = False

    while not done:
        preprocessed = preprocess_obs(obs, device)
        frame_stack[:, :3] = frame_stack[:, 1:].clone()
        frame_stack[:, 3] = preprocessed

        with torch.no_grad():
            dist, _ = agent.get_action_and_value(frame_stack)
            action = dist.mean.cpu().numpy()[0]

        obs_np, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = torch.from_numpy(obs_np.astype(np.uint8)).unsqueeze(0)

    # A mean episode reward of around 900 or higher is generally considered very good.
    # ~900–950: Strong performance, near-human level.
    # >900 consistently: Often cited as having “solved” CarRacing-v0/v1/v2/v3 in research and tutorials.
    print(f"Evaluation completed. Total reward: {total_reward:.2f}")
    env.close()


# ----------------------
# Args & Main
# ----------------------
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
    batch_size: int
    minibatch_size: int
    num_iterations: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--torch-deterministic", type=bool, default=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)  # Good for M1
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
