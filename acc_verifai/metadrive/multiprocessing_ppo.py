import contextlib
import logging
import pathlib
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp  # Use torch multiprocessing
from gymnasium import spaces
from torch import nn, optim

import scenic
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator


# --- Configuration ---
ENV_NAME = "IDM-MOBIL Distrupt"  # Environment to use
NUM_WORKERS = 4         # Number of parallel processes for data collection
TOTAL_TIMESTEPS = 1_000_000 # Total timesteps for training
STEPS_PER_WORKER = 256  # Timesteps collected by each worker per iteration
NUM_EPOCHS = 4          # Number of optimization epochs per PPO iteration
MINIBATCH_SIZE = 64      # Size of minibatches for optimization
GAMMA = 0.99             # Discount factor
GAE_LAMBDA = 0.95        # Lambda for Generalized Advantage Estimation
CLIP_EPSILON = 0.2       # PPO clipping parameter
LR = 3e-4                # Learning rate
ENTROPY_COEF = 0.01      # Entropy coefficient for exploration bonus
VALUE_LOSS_COEF = 0.5    # Value function loss coefficient
MAX_GRAD_NORM = 0.5      # Gradient clipping threshold
SEED = 42                # Random seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

LOG_STD_MAX = 2
LOG_STD_MIN = -5
# --- Actor-Critic Network ---
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
        self.critic = nn.Linear(hidden_dim, 1)       # Value head

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
             # Ensure input is a tensor, handle potential double type from numpy
            x = torch.tensor(x, dtype=torch.float32)
        shared_features = self.shared_layer(x)
        mean = self.fc_mean(shared_features)
        log_std = self.fc_logstd(shared_features)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        value = self.critic(shared_features)
        return mean, log_std, value

# --- Worker Function ---
def worker_fn(worker_id: int, steps_per_worker: int, model_state_dict: dict, data_queue: deque, seed: int) -> None:
    """Execute function for each worker process. Initializes environment and model, collects trajectories, and sends data back."""
    logger.debug("Worker %s: Initializing...", worker_id)
    # Each worker needs its own environment instance and random seed
    scenario = scenic.scenarioFromFile(
        "idm.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )
    env = ScenicGymEnv(
        scenario,
        MetaDriveSimulator(timestep=0.1, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=False, real_time=False),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(5,4)),
        action_space=spaces.Box(low=-1, high=1, shape=(2,)),
        max_steps=700,
    )
    obs_space_shape = env.observation_space.shape
    action_space = env.action_space
    # Ensure the observation space shape is correctly handled (e.g., flattened if needed)
    obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]

    # Seed the environment for reproducibility within the worker
    # Use worker_id and the main seed to ensure different seeds per worker
    worker_seed = seed + worker_id
    env.reset(seed=worker_seed)
    # Note: gym.make doesn't directly accept seed, reset does.

    # Initialize local model and load state dict
    local_model = ActorCritic(obs_dim, action_space)
    local_model.load_state_dict(model_state_dict)
    local_model.eval() # Set model to evaluation mode for rollouts

    # Storage for trajectory data
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()
    current_step = 0
    while current_step < steps_per_worker:
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0) # Flatten obs if needed

        with torch.no_grad(): # No need to track gradients during rollout
            mean, log_std, value = local_model(obs_tensor)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * local_model.action_scale + local_model.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(local_model.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

        action = action.cpu().numpy().squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition data
        observations.append(obs.flatten()) # Store flattened obs
        actions.append(action)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        obs = next_obs
        current_step += 1

        if done:
            obs, _ = env.reset() # Reset environment if episode ends

    # Calculate the value of the last state for GAE calculation
    last_obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, _, last_value = local_model(last_obs_tensor)
        last_value = last_value.item()

    # Convert lists to numpy arrays for efficient transfer
    trajectory_data = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
        "values": np.array(values, dtype=np.float32),
        "last_value": last_value,
        "last_done": done, # Needed for GAE calculation boundary
    }

    # Put data into the queue
    data_queue.put(trajectory_data)
    logger.debug("Worker %s: Finished collecting %s steps.", worker_id, current_step)
    env.close() # Clean up environment resources

# --- GAE Calculation ---
def compute_gae(
        rewards: np.array,
        values:np.array,
        dones:np.array,
        last_value: float,
        last_done: float,
        gamma: float,
        gae_lambda:float,
    ) -> tuple:
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    num_steps = len(rewards)
    next_values = np.append(values[1:], last_value if not last_done else 0.0) # Use last_value if not done
    next_non_terminal = 1.0 - dones # 1 if not done, 0 if done

    # Calculate TD errors (deltas)
    deltas = rewards + gamma * next_values * next_non_terminal - values

    # Calculate advantages using GAE formula, iterating backwards
    for t in reversed(range(num_steps)):
        last_gae_lam = deltas[t] + gamma * gae_lambda * next_non_terminal[t] * last_gae_lam
        advantages[t] = last_gae_lam

    # Calculate returns (targets for value function)
    returns = advantages + values
    return advantages, returns

EPSILON = 1e-5
# --- PPO Update Function ---
def ppo_update(model: nn.Module, optimizer: optim.Optimizer, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_log_probs_old: torch.Tensor,
               batch_advantages: torch.Tensor, batch_returns: torch.Tensor, num_epochs: int, minibatch_size: int,
               clip_epsilon: float, entropy_coef: float, value_loss_coef: float, max_grad_norm: float) -> None:
    """Perform the PPO update step using collected batch data."""
    batch_size = batch_obs.size(0)
    # Normalize advantages (important for stability)
    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

    for _epoch in range(num_epochs):
        # Shuffle data indices for minibatch creation
        indices = rng.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]

            # Get minibatch data
            mb_obs = batch_obs[minibatch_indices]
            mb_actions = batch_actions[minibatch_indices]
            mb_log_probs_old = batch_log_probs_old[minibatch_indices]
            mb_advantages = batch_advantages[minibatch_indices]
            mb_returns = batch_returns[minibatch_indices]

            # Get new log probabilities, values, and entropy from the current policy
            mean, log_std, values_pred = model(mb_obs)
            std = log_std.exp()

            normal = torch.distributions.Normal(mean, std)

            mb_actions_clamped = torch.clamp(mb_actions, -1.0 + EPSILON, 1.0 - EPSILON)
            unsquashed_mb_actions = torch.atanh(mb_actions_clamped)
            log_probs_gaussian = normal.log_prob(unsquashed_mb_actions).sum(dim=-1)
            log_prob_squash_correction = torch.log(1.0 - mb_actions.pow(2) + EPSILON).sum(dim=-1)
            log_probs_new = log_probs_gaussian - log_prob_squash_correction

            # Entropy of the Gaussian distribution (before squashing)
            entropy = normal.entropy().mean()
            values_pred = values_pred.squeeze(-1) # Ensure value shape matches returns

            # Calculate Policy Loss (Clipped Surrogate Objective)
            prob_ratio = torch.exp(log_probs_new - mb_log_probs_old)
            surr1 = prob_ratio * mb_advantages
            surr2 = torch.clamp(prob_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate Value Loss (MSE)
            # TODO: Optional: Clip value loss (often used in PPO implementations)
            value_loss = 0.5 * ((values_pred - mb_returns) ** 2).mean() # Simple MSE loss

            # Calculate Total Loss
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

# --- Main Training Loop ---
def main() -> None:
    """Run main function to set up and run the PPO training."""
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn") # 'spawn' is generally safer than 'fork'
        pass # Already set or not applicable

    logger.info("Starting PPO training...")
    logger.info("Environment: %s, Workers: %s, Total Timesteps: %s", ENV_NAME, NUM_WORKERS, TOTAL_TIMESTEPS)
    logger.info("Hyperparameters: gamma=%s, lambda=%s, clip_eps=%s, lr=%s", GAMMA, GAE_LAMBDA, CLIP_EPSILON, LR)

    # Create a dummy environment to get observation and action space dimensions
    obs_space_shape = (5,4)
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]

    # Initialize the Actor-Critic model and optimizer
    model = ActorCritic(obs_dim, action_space)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Calculate total number of updates
    batch_size = NUM_WORKERS * STEPS_PER_WORKER
    num_updates = TOTAL_TIMESTEPS // batch_size
    logger.info("Batch Size (Workers * Steps): %s", batch_size)
    logger.info("Total PPO Updates: %s", num_updates)

    # Queue for collecting data from workers
    data_queue = mp.Queue()

    # Performance tracking
    total_steps = 0
    start_time = time.time()
    episode_rewards = deque(maxlen=100) # Store rewards of last 100 episodes
    episode_lengths = deque(maxlen=100) # Store lengths of last 100 episodes
    total_episodes = 0

    # --- Training Loop ---
    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        model.eval() # Set model to evaluation mode for rollouts

        # --- Parallel Data Collection ---
        processes = []
        current_model_state_dict = model.state_dict() # Get current weights
        # Share state dict (serializable) instead of the whole model
        for i in range(NUM_WORKERS):
            p = mp.Process(target=worker_fn, args=(i, STEPS_PER_WORKER,
                                                   current_model_state_dict, data_queue, SEED + update * NUM_WORKERS)) # Pass unique seed component
            p.start()
            processes.append(p)

        # Collect data from the queue
        all_trajectory_data = [data_queue.get() for _ in range(NUM_WORKERS)]

        # Wait for all processes to finish
        for p in processes:
            p.join()
        logger.debug("Update %s: All workers finished.", update)

        # --- Process Collected Data ---
        batch_obs_list = []
        batch_actions_list = []
        batch_log_probs_list = []
        batch_advantages_list = []
        batch_returns_list = []

        # Track episode stats from collected data (approximate)
        for data in all_trajectory_data:
            # Calculate advantages and returns for this worker's trajectory
            advantages, returns = compute_gae(
                data["rewards"], data["values"], data["dones"],
                data["last_value"], data["last_done"], GAMMA, GAE_LAMBDA,
            )
            batch_advantages_list.append(advantages)
            batch_returns_list.append(returns)

            # Append other data
            batch_obs_list.append(data["observations"])
            batch_actions_list.append(data["actions"])
            batch_log_probs_list.append(data["log_probs"])

            # Track episode returns and lengths based on 'dones'
            current_episode_reward = 0
            current_episode_length = 0
            for reward, done in zip(data["rewards"], data["dones"]):  # noqa: B905
                current_episode_reward += reward
                current_episode_length += 1
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    total_episodes += 1
                    current_episode_reward = 0
                    current_episode_length = 0


        # Concatenate data from all workers into single batches
        batch_obs = torch.tensor(np.concatenate(batch_obs_list), dtype=torch.float32)
        batch_actions = torch.tensor(np.concatenate(batch_actions_list), dtype=torch.int64)
        batch_log_probs_old = torch.tensor(np.concatenate(batch_log_probs_list), dtype=torch.float32)
        batch_advantages = torch.tensor(np.concatenate(batch_advantages_list), dtype=torch.float32)
        batch_returns = torch.tensor(np.concatenate(batch_returns_list), dtype=torch.float32)

        # --- PPO Update Step ---
        model.train() # Set model to training mode
        ppo_update(model, optimizer, batch_obs, batch_actions, batch_log_probs_old,
                   batch_advantages, batch_returns, NUM_EPOCHS, MINIBATCH_SIZE,
                   CLIP_EPSILON, ENTROPY_COEF, VALUE_LOSS_COEF, MAX_GRAD_NORM)

        # --- Logging and Timing ---
        total_steps += batch_size
        update_end_time = time.time()
        fps = int(batch_size / (update_end_time - update_start_time))
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0

        if update % 10 == 0 or update == 1: # Log every 10 updates or the first one
             logger.info(
                "Update: %s/%s, Timesteps: %s/%s, FPS: %s, Episodes: %s, Avg Reward (Last 100): %.2f, Avg Length (Last 100): %.2f",
                update, num_updates, total_steps, TOTAL_TIMESTEPS, fps, total_episodes, avg_reward, avg_length,
             )

    # --- End of Training ---
    end_time = time.time()
    logger.info("Training finished in %.2f seconds.", end_time - start_time)

    # Optional: Save the trained model
    torch.save(model.state_dict(), f"ppo_{ENV_NAME}_model.pth")
    logger.info("Model saved to ppo_%s_model.pth", ENV_NAME)

if __name__ == "__main__":
    # This check is important for multiprocessing on Windows/MacOS
    main()
