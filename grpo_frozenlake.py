import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import RecordEpisodeStatistics
from dataclasses import dataclass, field
from typing import Callable


# =====================
# 1. Environment Wrappers
# =====================
class OneHotWrapper(gym.ObservationWrapper[np.ndarray, int, Discrete]):
    """Wrapper to convert discrete state to one-hot encoding"""

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


class StateResetWrapper(gym.Wrapper):
    """Wrapper to enable state setting/resetting for FrozenLake"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.saved_state = None

    def save_state(self):
        self.saved_state = self.unwrapped.s

    def reset_to_saved(self):
        self.unwrapped.s = self.saved_state
        return self.observation(self.saved_state)


# =====================
# 2. Policy Network
# =====================
class PolicyNetwork(nn.Module):
    """Policy-only network for GRPO"""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================
# 3. Data Structures
# =====================
@dataclass
class TrajectoryData:
    """Storage for rollout data"""

    states: list[int] = field(default_factory=list)  # State indices
    actions: list[int] = field(default_factory=list)  # Actions taken
    rewards: list[float] = field(default_factory=list)  # Immediate rewards
    logprobs: list[float] = field(default_factory=list)  # Log probabilities
    state_values: dict[int, float] = field(
        default_factory=dict
    )  # Mean reward per state


class Args(argparse.Namespace):
    """Hyperparameters for GRPO training"""

    # Training parameters
    seed: int
    torch_deterministic: bool
    total_timesteps: int
    num_envs: int
    num_steps_per_rollout: int
    gamma: float
    num_minibatches: int
    epochs: int
    learning_rate: float
    clip_coef: float
    entropy_coef: float
    max_grad_norm: float

    # GRPO-specific parameters
    num_return_samples: int  # Multiple rewards per state
    num_bins: int  # For group normalization

    # Environment parameters
    env_id: str
    is_slippery: bool

    # Calculated during setup
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# =====================
# 4. Environment Setup
# =====================
def make_raw_env(
    env_id: str, is_slippery: bool = False, render_mode: str | None = None
):
    return gym.make(env_id, is_slippery=is_slippery, render_mode=render_mode)


def make_env(env_id: str, is_slippery: bool) -> Callable[[], gym.Env]:
    """Environment factory function with state reset capability"""

    def thunk():
        env = make_raw_env(env_id, is_slippery)
        env = StateResetWrapper(env)
        env = RecordEpisodeStatistics(env)
        env = OneHotWrapper(env)
        return env

    return thunk


# =====================
# 5. GRPO Core Functions
# =====================
def sample_state_rewards(
    state_idx: int,
    policy: PolicyNetwork,
    num_samples: int,
    make_env_func: Callable,
    device: torch.device,
) -> list[float]:
    """
    Sample multiple immediate rewards from a single state
    by taking different actions with the current policy
    """
    # Create environment and set to target state
    env = make_env_func()
    env.save_state()
    env.unwrapped.s = state_idx

    rewards = []
    obs = env.observation(env.unwrapped.s)

    for _ in range(num_samples):
        # Convert to tensor and get action
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample().item()

        # Take ONE STEP ONLY
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        # Reset to original state
        env.reset_to_saved()

    env.close()
    return rewards


def calculate_group_normalized_advantages(
    trajectory: TrajectoryData, num_bins: int
) -> list[float]:
    """
    Calculate group-normalized advantages:
    1. Compute relative advantage per state: reward - mean_reward
    2. Group states by mean_reward into bins
    3. Normalize advantages within each bin
    """
    # Step 1: Calculate raw relative advantages
    advantages = []
    for state, reward in zip(trajectory.states, trajectory.rewards):
        mean_reward = trajectory.state_values[state]
        advantages.append(reward - mean_reward)

    # Step 2: Group by state value (mean_reward)
    bin_values = np.array(list(trajectory.state_values.values()))
    bin_edges = np.linspace(bin_values.min(), bin_values.max(), num_bins + 1)

    # Create bins
    bins = [[] for _ in range(num_bins)]
    for i, (state, value) in enumerate(trajectory.state_values.items()):
        bin_idx = np.digitize(value, bin_edges) - 1
        bin_idx = max(0, min(bin_idx, num_bins - 1))  # Clamp to valid range
        bins[bin_idx].append(i)

    # Step 3: Normalize within bins
    bin_means = []
    bin_stds = []
    for bin_indices in bins:
        if bin_indices:
            bin_adv = [advantages[i] for i in bin_indices]
            bin_means.append(np.mean(bin_adv))
            bin_stds.append(np.std(bin_adv) or 1.0)  # Avoid division by zero
        else:
            bin_means.append(0)
            bin_stds.append(1)

    # Apply normalization
    normalized_advantages = []
    for i, adv in enumerate(advantages):
        for bin_idx, bin_indices in enumerate(bins):
            if i in bin_indices:
                normalized_advantages.append(
                    (adv - bin_means[bin_idx]) / bin_stds[bin_idx]
                )
                break
        else:  # If not found in any bin (shouldn't happen)
            normalized_advantages.append(adv)

    return normalized_advantages


# =====================
# 6. Training Loop
# =====================
def train(args: "Args"):
    # Set up logging and device
    writer = SummaryWriter(f"runs/grpo_{args.env_id.lower()}_{int(time.time())}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.is_slippery) for _ in range(args.num_envs)]
    )
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.n

    # Initialize policy
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize variables
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    # Main training loop
    for iteration in range(args.num_iterations):
        print(f"Iteration: {iteration}/{args.num_iterations}")
        trajectory = TrajectoryData()

        # Rollout collection
        for step in range(args.num_steps_per_rollout):
            # Convert one-hot state back to index (for FrozenLake)
            state_indices = [
                torch.argmax(next_obs[i]).item() for i in range(args.num_envs)
            ]

            with torch.no_grad():
                logits = policy(next_obs)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            # Execute environment step
            next_obs, rewards, terminated, truncated, infos = envs.step(
                actions.cpu().numpy()
            )
            next_done = np.logical_or(terminated, truncated)

            # Store transition
            for i in range(args.num_envs):
                trajectory.states.append(state_indices[i])
                trajectory.actions.append(actions[i].item())
                trajectory.rewards.append(rewards[i])
                trajectory.logprobs.append(logprobs[i].item())

                # Log episode completion
                if next_done[i] and "episode" in infos:
                    print(
                        f"Global Step: {global_step}, Reward: {infos['episode']['r'][i]}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", infos["episode"]["r"][i], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", infos["episode"]["l"][i], global_step
                    )

            # Prepare next iteration
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.tensor(next_done, device=device)
            global_step += args.num_envs

        # ================================================
        # GRPO: Sample multiple rewards per state
        # ================================================
        unique_states = set(trajectory.states)
        print(f"Sampling rewards for {len(unique_states)} unique states...")

        # Calculate mean reward for each state
        for state_idx in unique_states:
            rewards = sample_state_rewards(
                state_idx=state_idx,
                policy=policy,
                num_samples=args.num_return_samples,
                make_env_func=make_env(args.env_id, args.is_slippery),
                device=device,
            )
            trajectory.state_values[state_idx] = np.mean(rewards)

        # Calculate group-normalized advantages
        advantages = calculate_group_normalized_advantages(trajectory, args.num_bins)
        advantages = torch.tensor(advantages, device=device, dtype=torch.float32)

        # Convert trajectory data to tensors
        states_tensor = torch.tensor(
            [np.eye(obs_dim)[s] for s in trajectory.states],  # One-hot encoding
            device=device,
            dtype=torch.float32,
        )
        actions_tensor = torch.tensor(
            trajectory.actions, device=device, dtype=torch.long
        )
        old_logprobs = torch.tensor(
            trajectory.logprobs, device=device, dtype=torch.float32
        )

        # Policy optimization
        policy_losses = []
        entropy_losses = []

        for epoch in range(args.epochs):
            # Shuffle indices for minibatch updates
            indices = np.arange(len(trajectory.states))
            np.random.shuffle(indices)

            # Minibatch updates
            for start in range(0, len(indices), args.minibatch_size):
                end = start + args.minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_states = states_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Get new policy distribution
                logits = policy(mb_states)
                dist = Categorical(logits=logits)
                mb_new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss with clipping
                ratio = (mb_new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -entropy

                # Total loss
                loss = policy_loss + args.entropy_coef * entropy_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                # Store losses for logging
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy.item())

        # Log training metrics
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_entropy = np.mean(entropy_losses) if entropy_losses else 0

        writer.add_scalar("losses/policy_loss", avg_policy_loss, global_step)
        writer.add_scalar("losses/entropy", avg_entropy, global_step)

        # Performance stats
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # Clean up
    print("Training complete. Saving model...")
    torch.save(policy.state_dict(), f"grpo_{args.env_id.lower()}.pt")
    envs.close()
    writer.close()


# =====================
# 7. Evaluation
# =====================
def evaluate(args: "Args"):
    # Create environment
    env = make_raw_env(args.env_id, args.is_slippery, render_mode="human")
    env = StateResetWrapper(env)
    env = RecordEpisodeStatistics(env)
    env = OneHotWrapper(env)

    # Load trained policy
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    policy.load_state_dict(torch.load(f"grpo_{args.env_id.lower()}.pt"))
    policy.eval()

    # Run evaluation
    obs, _ = env.reset()
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    # Parse and validate arguments
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--torch-deterministic", type=bool, default=True)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-steps-per-rollout", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.08)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # GRPO-specific parameters
    parser.add_argument("--num-return-samples", type=int, default=8)
    parser.add_argument("--num-bins", type=int, default=10)

    # Environment parameters
    parser.add_argument("--env-id", type=str, default="FrozenLake-v1")
    parser.add_argument("--is-slippery", type=bool, default=False)

    # Calculate runtime parameters
    args = parser.parse_args(namespace=Args)
    args.batch_size = int(args.num_envs * args.num_steps_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Run training and evaluation
    train(args)
    evaluate(args)
