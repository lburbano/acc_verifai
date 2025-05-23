# %%
import contextlib
import logging
import pathlib
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import scenic
import torch
import torch.multiprocessing as mp
import tyro
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from torch import nn, optim


device = "cuda" if torch.cuda.is_available() else "cpu"

# %%


@dataclass
class Args:
    """Hyperparameters and configuration for the PPO training."""

    # Environment/scenic file to use
    scenic_file: str = "idm.scenic"
    # Number of parallel processes for data collection
    num_workers: int = 4
    # Total timesteps for training
    total_timesteps: int = 1_000_000
    # Timesteps collected by each worker per iteration
    steps_per_worker: int = 256
    # Number of optimization epochs per PPO iteration
    num_epochs: int = 4
    # Size of minibatches for optimization
    minibatch_size: int = 6
    # Discount factor
    gamma: float = 0.9
    # Lambda for Generalized Advantage Estimation
    gae_lambda: float = 0.9
    # PPO clipping parameter
    clip_epsilon: float = 0.2
    # Learning rate
    lr: float = 3e-4
    # Entropy coefficient for exploration bonus
    entropy_coef: float = 0.01
    # Value function loss coefficient
    value_loss_coef: float = 0.5
    # Gradient clipping threshold
    max_grad_norm: float = 0.5
    # Random seed
    seed: int = 4
    # Directory to save models
    model_dir: str = "models"


LOG_STD_MAX = 2
LOG_STD_MIN = -5
EPSILON = 1e-5


class ActorCritic(nn.Module):
    """A simple Actor-Critic network for discrete action spaces. Shares layers between actor and critic."""

    def __init__(self, obs_dim: int, action_space: spaces.Box, hidden_dim: int = 64):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.fc_mean = nn.Linear(hidden_dim, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_space.shape))
        self.critic = nn.Linear(hidden_dim, 1)  # Value head

        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x: any) -> tuple:
        """Forward pass through the network."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        shared_features = self.shared_layer(x)
        mean = self.fc_mean(shared_features)
        log_std = self.fc_logstd(shared_features)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        value = self.critic(shared_features)
        return mean, log_std, value


# %%
def worker_fn(worker_id: int, steps_per_worker: int, model_state_dict: dict, data_queue: mp.Queue, seed: int, scenic_file: str) -> None:
    """Execute function for each worker process. Initializes environment and model, collects trajectories, and sends data back."""
    logging.getLogger(__name__).debug("Worker %s: Initializing...", worker_id)
    scenario = scenic.scenarioFromFile(
        scenic_file,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )
    env = ScenicGymEnv(
        scenario,
        MetaDriveSimulator(timestep=0.05, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=False, real_time=False),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(5, 7)),
        action_space=spaces.Box(low=-1, high=1, shape=(2,)),
        max_steps=700,
    )
    obs_space_shape = env.observation_space.shape
    action_space = env.action_space
    obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]
    worker_seed = seed + worker_id
    env.reset(seed=worker_seed)

    local_model = ActorCritic(obs_dim, action_space)
    local_model.load_state_dict(model_state_dict)
    local_model.eval()

    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()
    current_step = 0
    while current_step < steps_per_worker:
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, log_std, value = local_model(obs_tensor)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * local_model.action_scale + local_model.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(local_model.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        action = action.cpu().numpy().squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(obs.flatten())
        actions.append(action)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        obs = next_obs
        current_step += 1

        if done:
            obs, _ = env.reset()

    last_obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, _, last_value = local_model(last_obs_tensor)
        last_value = last_value.item()

    trajectory_data = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
        "values": np.array(values, dtype=np.float32),
        "last_value": last_value,
        "last_done": done,
    }

    data_queue.put(trajectory_data)
    logging.getLogger(__name__).debug("Worker %s: Finished collecting %s steps.", worker_id, current_step)
    env.close()


# %%
def compute_gae(
    rewards: np.array,
    values: np.array,
    dones: np.array,
    last_value: float,
    last_done: float,
    gamma: float,
    gae_lambda: float,
) -> tuple:
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    num_steps = len(rewards)
    next_values = np.append(values[1:], last_value if not last_done else 0.0)
    next_non_terminal = 1.0 - dones
    deltas = rewards + gamma * next_values * next_non_terminal - values

    for t in reversed(range(num_steps)):
        last_gae_lam = deltas[t] + gamma * gae_lambda * next_non_terminal[t] * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


# %%
def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch_obs: torch.Tensor,
    batch_actions: torch.Tensor,
    batch_log_probs_old: torch.Tensor,
    batch_advantages: torch.Tensor,
    batch_returns: torch.Tensor,
    num_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
    rng: np.random.Generator,
) -> None:
    """Perform the PPO update step using collected batch data."""
    batch_size = batch_obs.size(0)
    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

    for _ in range(num_epochs):
        indices = rng.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]

            mb_obs = batch_obs[minibatch_indices]
            mb_actions = batch_actions[minibatch_indices]
            mb_log_probs_old = batch_log_probs_old[minibatch_indices]
            mb_advantages = batch_advantages[minibatch_indices]
            mb_returns = batch_returns[minibatch_indices]

            mean, log_std, values_pred = model(mb_obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)

            mb_actions_clamped = torch.clamp(mb_actions, -1.0 + EPSILON, 1.0 - EPSILON)
            unsquashed_mb_actions = torch.atanh(mb_actions_clamped)
            log_probs_gaussian = normal.log_prob(unsquashed_mb_actions).sum(dim=-1)
            log_prob_squash_correction = torch.log(1.0 - mb_actions.pow(2) + EPSILON).sum(dim=-1)
            log_probs_new = log_probs_gaussian - log_prob_squash_correction

            entropy = normal.entropy().mean()
            values_pred = values_pred.squeeze(-1)

            prob_ratio = torch.exp(log_probs_new - mb_log_probs_old)
            surr1 = prob_ratio * mb_advantages
            surr2 = torch.clamp(prob_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((values_pred - mb_returns) ** 2).mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def main() -> None:
    """Run the PPO training."""
    args = tyro.cli(Args)

    # Ensure model directory exists
    if not pathlib.Path.exists(pathlib.Path(args.model_dir)):
        pathlib.Path.mkdir(pathlib.Path(args.model_dir))
    print("Model directory:", args.model_dir)

    # Set the environment name based on the scenic file
    env_name = pathlib.Path(args.scenic_file).stem

    # Set up multiprocess start method
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")

    # seeds
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting PPO training...")
    logger.info("Environment: %s, Workers: %s, Total Timesteps: %s", env_name, args.num_workers, args.total_timesteps)
    logger.info("Hyperparameters: gamma=%s, lambda=%s, clip_eps=%s, lr=%s", args.gamma, args.gae_lambda, args.clip_epsilon, args.lr)

    # temp env to get obs and action space
    env = ScenicGymEnv(
        env_name,
        MetaDriveSimulator(timestep=0.05, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=False, real_time=False),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(5, 7)),
        action_space=spaces.Box(low=-1, high=1, shape=(2,)),
        max_steps=700,
    )
    obs_space_shape = env.observation_space.shape
    action_space = env.action_space
    obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]
    env.close()

    model = ActorCritic(obs_dim, action_space).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    batch_size = args.num_workers * args.steps_per_worker
    num_updates = args.total_timesteps // batch_size
    logger.info("Batch Size (Workers * Steps): %s", batch_size)
    logger.info("Total PPO Updates: %s", num_updates)

    data_queue = mp.Queue()

    total_steps = 0
    start_time = time.time()
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    total_episodes = 0

    # Training Loop
    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        model.eval()

        processes = []
        current_model_state_dict = model.state_dict()
        for i in range(args.num_workers):
            p = mp.Process(
                target=worker_fn,
                args=(
                    i,
                    args.steps_per_worker,
                    current_model_state_dict,
                    data_queue,
                    args.seed + update * args.num_workers,
                    args.scenic_file,
                ),
            )
            p.start()
            processes.append(p)

        all_trajectory_data = [data_queue.get() for _ in range(args.num_workers)]
        for p in processes:
            p.join()
        logger.debug("Update %s: All workers finished.", update)

        batch_obs_list = []
        batch_actions_list = []
        batch_log_probs_list = []
        batch_advantages_list = []
        batch_returns_list = []

        for data in all_trajectory_data:
            advantages, returns = compute_gae(
                data["rewards"],
                data["values"],
                data["dones"],
                data["last_value"],
                data["last_done"],
                args.gamma,
                args.gae_lambda,
            )
            batch_advantages_list.append(advantages)
            batch_returns_list.append(returns)
            batch_obs_list.append(data["observations"])
            batch_actions_list.append(data["actions"])
            batch_log_probs_list.append(data["log_probs"])

            current_episode_reward = 0
            current_episode_length = 0
            for reward, done in zip(data["rewards"], data["dones"], strict=False):
                current_episode_reward += reward
                current_episode_length += 1
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    total_episodes += 1
                    current_episode_reward = 0
                    current_episode_length = 0

        batch_obs = torch.tensor(np.concatenate(batch_obs_list), dtype=torch.float32).to(device)
        batch_actions = torch.tensor(np.concatenate(batch_actions_list), dtype=torch.float32).to(device)
        batch_log_probs_old = torch.tensor(np.concatenate(batch_log_probs_list), dtype=torch.float32).to(device)
        batch_advantages = torch.tensor(np.concatenate(batch_advantages_list), dtype=torch.float32).to(device)
        batch_returns = torch.tensor(np.concatenate(batch_returns_list), dtype=torch.float32).to(device)

        model.train()
        ppo_update(
            model,
            optimizer,
            batch_obs,
            batch_actions,
            batch_log_probs_old,
            batch_advantages,
            batch_returns,
            args.num_epochs,
            args.minibatch_size,
            args.clip_epsilon,
            args.entropy_coef,
            args.value_loss_coef,
            args.max_grad_norm,
            rng,
        )

        total_steps += batch_size
        update_end_time = time.time()
        fps = int(batch_size / (update_end_time - update_start_time))
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0

        if update % 1 == 0 or update == 1:
            logger.info(
                "Update: %s/%s, Timesteps: %s/%s, FPS: %s, Episodes: %s, Avg Reward (Last 100): %.2f, Avg Length (Last 100): %.2f",
                update,
                num_updates,
                total_steps,
                args.total_timesteps,
                fps,
                total_episodes,
                avg_reward,
                avg_length,
            )
            # Save model every 10 updates
            torch.save(model.state_dict(), f"{args.model_dir}/ppo_{env_name}_model.pth")

    end_time = time.time()
    logger.info("Training finished in %.2f seconds.", end_time - start_time)

    torch.save(model.state_dict(), f"{args.model_dir}/ppo_{env_name}_model.pth")
    logger.info("Model saved to ppo_%s_model.pth", env_name)


if __name__ == "__main__":
    main()
