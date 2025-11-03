import argparse
import random
import time
from typing import Callable, cast
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordEpisodeStatistics
from dataclasses import dataclass

from utils.paths import RUNS_DIR


# ----------------------
# 1. One-Hot Wrapper
# ----------------------
class OneHotWrapper(gym.ObservationWrapper[np.ndarray, int, Discrete]):
    def __init__(self, env: Env[Discrete, int], num_states: int):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=1, shape=(num_states,), dtype=np.float32
        )

    def observation(self, obs: np.int64) -> np.ndarray:
        assert isinstance(self.env.observation_space, Discrete), (
            "OneHotWrapper intended to work with Discrete observation space"
        )
        one_hot = np.zeros(self.env.observation_space.n)
        one_hot[obs] = 1.0
        return one_hot


# ----------------------
# 2. Agent
# ----------------------
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def get_action(self, x):
        x = self.net(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# ----------------------
# 3. Dataclass
# ----------------------
@dataclass
class ProcessedRollout:
    """Processed data ready for training."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    # rewards: torch.Tensor
    # dones: torch.Tensor


# ----------------------
# 4. Training Loop
# ----------------------
def make_raw_env(
    env_id: str, is_slippery: bool = False, render_mode: str | None = None
):
    return gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode=render_mode)


def make_env(env_id: str, is_slippery: bool) -> Callable[[], Env[np.ndarray, int]]:
    def thunk():
        env = cast(Env[Discrete, int], make_raw_env(env_id, is_slippery))
        # env = cast(Env[Discrete, int], gym.make("CliffWalking-v1", is_slippery=False))
        env = RecordEpisodeStatistics(env)
        env = OneHotWrapper(
            env, num_states=int(cast(Discrete, env.observation_space).n)
        )  # 48 states in CliffWalking
        return env

    return thunk


def train(args: "Args"):
    writer = SummaryWriter(
        f"{RUNS_DIR}/frozenlake/ppo_scratch_{args.env_id.lower()}_{int(time.time())}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Create environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.is_slippery) for _ in range(args.num_envs)],
    )
    # since SyncVectorEnv definition loses the observation space and action space types
    # we might as well cast()
    obs_space = cast(Box, envs.single_observation_space)
    act_space = cast(Discrete, envs.single_action_space)

    obs_dim = obs_space.shape[0]  # 48
    act_dim = act_space.n.item()  # 4

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs = torch.zeros(
        # it works like (64, 2) + (4,) => (128, 4, 4)
        (args.num_steps_per_rollout, args.num_envs) + obs_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps_per_rollout, args.num_envs)
        + cast(tuple[int, ...], act_space.shape)
    ).to(device)

    logprobs = torch.zeros((args.num_steps_per_rollout, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps_per_rollout, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps_per_rollout, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps_per_rollout, args.num_envs)).to(device)

    # Initialize variables
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(args.num_iterations):
        # Collect rollout
        print("Iteration: %s", iteration)

        for step in range(0, args.num_steps_per_rollout):
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                logits, value = agent.get_action(next_obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = dist.log_prob(action)

            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminated, truncated)
            # done = torch.tensor(
            #     [t or d for t, d in zip(terminated, truncated)], device=device
            # )

            if np.any(next_done):
                # due to default autorest_mode=NEXT_STEP in gymnasium,
                # we have to location the exact done env index
                # refer to https://farama.org/Vector-Autoreset-Mode
                for i in range(args.num_envs):
                    if next_done[i]:
                        print(
                            f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}, episodic_length={infos['episode']['l'][i]}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return",
                            infos["episode"]["r"][i],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            infos["episode"]["l"][i],
                            global_step,
                        )
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            rewards[step] = torch.tensor(reward, device=device).view(-1)

            global_step += args.num_envs
            print("global_step: ", global_step)

        # Calculate returns and advantages
        with torch.no_grad():
            # calculate V_t+1 one more time to do the GAE part
            next_value = agent.get_action(torch.Tensor(next_obs).to(device))[
                1
            ].squeeze() * (1 - next_done)
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

        # it's now like a flatten list to do mini batch update the network weights and biases
        processed = ProcessedRollout(
            obs=obs.reshape((-1,) + obs_space.shape),
            actions=actions.reshape((-1,) + cast(tuple[int, ...], act_space.shape)),
            logprobs=logprobs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=values.reshape(-1),
        )
        # Optimize policy
        for _ in range(args.epochs):
            indices = np.random.permutation(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_indices = indices[start:end]

                p_obs = processed.obs[mb_indices]
                p_actions = processed.actions[mb_indices]
                old_logprobs = processed.logprobs[mb_indices]
                p_advantages = processed.advantages[mb_indices]
                p_returns = processed.returns[mb_indices]

                logits, p_values = agent.get_action(p_obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(p_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - old_logprobs).exp()

                mb_advantages = (p_advantages - p_advantages.mean()) / (
                    p_advantages.std()
                    + 1e-8  # just make sure that this is not divided by 0
                )

                # surrogate for surr
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    * mb_advantages
                )
                policy_loss = torch.min(surr1, surr2).mean()
                # value_loss = 0.5 * (returns_mb - values).pow(2).mean()
                newvalue = p_values.view(-1)
                v_loss_unclipped = (newvalue - p_returns) ** 2
                v_clipped = p_returns + torch.clamp(
                    newvalue - p_returns,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - p_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                # J(θ)=L_CLIP −c_1 * L_VF + c_2 * S
                # !!! Important !!!
                # The entropy coefficient in the loss function is currently set to 0.01.
                # A low entropy coefficient causes the policy to become deterministic too quickly,
                # reducing exploration.
                # So raise it from 0.01 to 0.05 or above
                target = (
                    policy_loss
                    - args.vf_coef * value_loss
                    + args.ent_coef * entropy_loss
                )
                # Loss = -J(θ)
                loss = -target
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Log metrics
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
    print("Training complete. Saving model...")
    torch.save(agent.state_dict(), f"ppo_{args.env_id.lower()}.pt")
    envs.close()
    writer.close()


# ----------------------
# 5. Evaluate Trained Policy
# ----------------------
def evaluate(args: "Args"):
    env = make_raw_env(args.env_id, args.is_slippery, render_mode="human")
    num_states = int(cast(Discrete, env.observation_space).n)
    num_actions = int(cast(Discrete, env.action_space).n)
    env = OneHotWrapper(env, num_states=num_states)
    policy = Agent(num_states, num_actions)
    policy.load_state_dict(torch.load(f"ppo_{args.env_id.lower()}.pt"))
    policy.eval()

    obs, _ = env.reset()
    done = False
    while not done:
        obs_tensor = torch.Tensor(obs).unsqueeze(0)
        logits, _ = policy.get_action(obs_tensor)
        action = Categorical(logits=logits).sample().item()
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated


class Args(argparse.Namespace):
    """Hyperparameters for PPO training."""

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

    # env specific
    env_id: str
    is_slippery: bool

    # calculation on runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--torch-deterministic", type=bool, default=True)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-steps-per-rollout", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.08)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--env-id", type=str, default="FrozenLake-v1")
    parser.add_argument("--is-slippery", type=bool, default=False)

    # below are calculated
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=0)

    args = parser.parse_args(namespace=Args())
    args.batch_size = int(args.num_envs * args.num_steps_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    train(args)
    evaluate(args)
