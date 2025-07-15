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
from dataclasses import dataclass, field


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
# 3. Dataclasses
# ----------------------
@dataclass
class Args:
    """Hyperparameters for PPO training."""

    seed: int = 2025
    torch_deterministic: bool = True
    total_timesteps: int = 100_000
    num_envs: int = 1
    num_steps_per_rollout: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.9
    num_minibatches: int = 4
    epochs: int = 8
    learning_rate: float = 2.5e-4
    clip_coef: float = 0.2
    max_grad_norm: float = 0.5
    # calculation on runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


@dataclass
class Rollout:
    """Collected data during a rollout."""

    obs: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    logprobs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)


@dataclass
class ProcessedRollout:
    """Processed data ready for training."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


# ----------------------
# 4. Training Loop
# ----------------------
def make_env() -> Callable[[], Env[np.ndarray, int]]:
    def thunk():
        env = cast(Env[Discrete, int], gym.make("CliffWalking-v1"))
        env = RecordEpisodeStatistics(env)
        env = OneHotWrapper(
            env, num_states=int(cast(Discrete, env.observation_space).n)
        )  # 48 states in CliffWalking
        return env

    return thunk


def train():
    args = Args()
    args.batch_size = int(args.num_envs * args.num_steps_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    writer = SummaryWriter(f"runs/ppo_cliffwalking_{int(time.time())}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Create environment
    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(args.num_envs)])
    # since SyncVectorEnv definition loses the observation space and action space types
    # we might as well cast()
    obs_space = cast(Box, envs.single_observation_space)
    act_space = cast(Discrete, envs.single_action_space)

    obs_dim = obs_space.shape[0]  # 48
    act_dim = act_space.n.item()  # 4

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize variables
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(args.num_iterations):
        # Collect rollout
        print("Iteration: %s", iteration)
        rollout = Rollout()

        for _ in range(0, args.num_steps_per_rollout):
            obs = torch.Tensor(next_obs).to(device)
            # done = next_done

            with torch.no_grad():
                logits, value = agent.get_action(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            # done = torch.tensor(
            #     [t or d for t, d in zip(terminated, truncated)], device=device
            # )

            rollout.obs.append(obs)
            rollout.actions.append(action)
            rollout.logprobs.append(dist.log_prob(action))
            rollout.rewards.append(torch.tensor(reward, device=device))
            next_done = torch.tensor(
                done, dtype=torch.float32, device=device
            )  # Convert to tensor
            rollout.dones.append(next_done)
            rollout.values.append(value)

            global_step += args.num_envs
            print("global_step: ", global_step)
            if "episode" in infos:
                print(
                    f"global_step={global_step}, episodic_return={infos['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", infos["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", infos["episode"]["l"], global_step
                )

        # Convert to processed tensors
        processed = ProcessedRollout(
            obs=torch.cat(rollout.obs),
            actions=torch.cat(rollout.actions),
            logprobs=torch.cat(rollout.logprobs),
            rewards=torch.cat(rollout.rewards),
            dones=torch.cat(rollout.dones),
            values=torch.cat(rollout.values),
        )

        # Calculate returns and advantages
        with torch.no_grad():
            # calculate V_t+1 one more time to do the GAE part
            next_value = agent.get_action(torch.Tensor(next_obs).to(device))[
                1
            ].squeeze() * (1 - next_done)
            advantages = torch.zeros_like(processed.rewards, device=device)
            returns = torch.zeros_like(processed.rewards, device=device)

            gae = 0
            for t in reversed(range(len(processed.rewards))):
                mask = 1 - processed.dones[t]
                delta = (
                    processed.rewards[t]
                    + args.gamma * next_value * mask
                    - processed.values[t]
                )
                gae = delta + args.gamma * args.gae_lambda * mask * gae
                advantages[t] = gae
                returns[t] = gae + processed.values[t]
                next_value = processed.values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy
        for _ in range(args.epochs):
            indices = np.random.permutation(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_indices = indices[start:end]

                obs = processed.obs[mb_indices]
                actions = processed.actions[mb_indices]
                old_logprobs = processed.logprobs[mb_indices]
                advantages_mb = advantages[mb_indices]
                returns_mb = returns[mb_indices]

                logits, values = agent.get_action(obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - old_logprobs).exp()

                advantages_mb = (advantages_mb - advantages_mb.mean()) / (
                    advantages_mb.std()
                    + 1e-8  # just make sure that this is not divided by 0
                )

                # surrogate for surr
                surr1 = ratio * advantages_mb
                surr2 = (
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    * advantages_mb
                )
                policy_loss = torch.min(surr1, surr2).mean()
                # value_loss = 0.5 * (returns_mb - values).pow(2).mean()
                newvalue = values.view(-1)
                v_loss_unclipped = (newvalue - returns_mb) ** 2
                v_clipped = returns_mb + torch.clamp(
                    newvalue - returns_mb,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns_mb) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                # J(θ)=L_CLIP −c_1 * L_VF + c_2 * S
                # !!! Important !!!
                # The entropy coefficient in the loss function is currently set to 0.01.
                # A low entropy coefficient causes the policy to become deterministic too quickly,
                # reducing exploration.
                # So raise it from 0.01 to 0.05 or above
                target = policy_loss - 0.5 * value_loss + 0.08 * entropy_loss
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
    torch.save(agent.state_dict(), "ppo_cliffwalking.pt")
    envs.close()
    writer.close()


# ----------------------
# 5. Evaluate Trained Policy
# ----------------------
def evaluate():
    env = gym.make("CliffWalking-v1", render_mode="human")
    env = OneHotWrapper(env, num_states=48)
    policy = Agent(48, 4)
    policy.load_state_dict(torch.load("ppo_cliffwalking.pt"))
    policy.eval()

    obs, _ = env.reset()
    done = False
    while not done:
        obs_tensor = torch.Tensor(obs).unsqueeze(0)
        logits, _ = policy.get_action(obs_tensor)
        action = Categorical(logits=logits).sample().item()
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated


if __name__ == "__main__":
    train()
    evaluate()
