import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from gymnasium import spaces
from torch import nn

import scenic
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Constants from Training Script (for action squashing) ---
LOG_STD_MAX = 2
LOG_STD_MIN = -5
EPSILON = 1e-5


# --- Model Definition (Copied directly from your training script) ---
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
        # logstd and critic are not strictly needed for deterministic inference, but defined for loading state_dict
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_space.shape))
        self.critic = nn.Linear(hidden_dim, 1)

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
            # Ensure input is tensor and on correct device
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
        elif torch.is_tensor(x):
            x = x.to(next(self.parameters()).device)

        shared_features = self.shared_layer(x)
        mean = self.fc_mean(shared_features)
        # Return only mean for deterministic action, placeholders for others if needed elsewhere
        return mean, None, None # mean, log_std, value

    def get_deterministic_action(self, x: any) -> np.ndarray:
        """Get the deterministic action (mean of the policy distribution)."""
        with torch.no_grad():
            mean, _, _ = self.forward(x)
            # Apply tanh squashing to the mean, then scale and bias
            y_t = torch.tanh(mean)
            action = y_t * self.action_scale + self.action_bias
        return action.cpu().numpy().squeeze(0) # Return as numpy array


# --- Argument Parsing ---
@dataclass
class InferenceArgs:
    """Configuration for running inference."""

    # Scenic file used during training
    scenic_file: str = "lane_follow.scenic" # Make sure this matches training
    # Path to the saved model file
    model_path: str = "models/ppo_lane_follow_model.pth" # Default name from training script
    # Number of episodes to run
    num_episodes: int = 10
    # Random seed for environment initialization (optional)
    seed: int = 42
    # Maximum steps per episode (should match training env setting if possible)
    max_steps: int = 700
    # Optional: Path to the SUMO map file if needed by scenic file/metadrive
    sumo_map_path: str = "../maps/Town06.net.xml" # Make sure this path is correct


def run_inference() -> None:
    """Load a trained model and run inference in the Scenic/MetaDrive environment."""
    args = tyro.cli(InferenceArgs)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting inference...")
    logger.info("Loading Scenic file: %s", args.scenic_file)
    logger.info("Loading Model from: %s", args.model_path)
    logger.info("Running for %s episodes.", args.num_episodes)

    # --- Initialize Environment ---
    try:
        scenario = scenic.scenarioFromFile(
            args.scenic_file,
            model="scenic.simulators.metadrive.model",
            mode2D=True,
        )
        # IMPORTANT: Set render=True for visualization
        # Set real_time=True if you want it to run closer to real speed, False for faster sim
        sumo_map_path = pathlib.Path(args.sumo_map_path) if args.sumo_map_path else None
        simulator = MetaDriveSimulator(
            timestep=0.1, sumo_map=sumo_map_path, render=True, real_time=False,
        )

        # Define observation and action spaces explicitly matching training
        # TODO: It's safer to get these from the training args/config if possible
        # These values MUST match the ones used during training
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5))
        act_space = spaces.Box(low=-1, high=1, shape=(1,))

        env = ScenicGymEnv(
            scenario,
            simulator,
            observation_space=obs_space,
            action_space=act_space,
            max_steps=args.max_steps,
        )
    except Exception:
        logger.exception("Failed to initialize environment")
        return

    # --- Initialize Model ---
    obs_dim = np.prod(obs_space.shape)
    model = ActorCritic(obs_dim, act_space).to(device)

    # --- Load Model Weights ---
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        logger.info("Model weights loaded successfully.")
    except FileNotFoundError:
        logger.exception("Model file not found at: %s", args.model_path)
        env.close()
        return
    except Exception:
        logger.exception("Failed to load model weights.")
        env.close()
        return

    # --- Inference Loop ---
    for episode in range(args.num_episodes):
        logger.info("Starting Episode %s...", episode + 1)
        obs, info = env.reset(seed=args.seed + episode) # Use different seed per episode if desired
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not terminated and not truncated:
            # Prepare observation tensor
            # Ensure obs is flattened correctly and has batch dim
            obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

            # Get deterministic action from the model
            action = model.get_deterministic_action(obs_tensor)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

        logger.info(
            "Episode %s finished. Reward: %.2f, Length: %s",
            episode + 1, episode_reward, episode_length,
        )

    logger.info("Inference finished.")
    env.close()


if __name__ == "__main__":
    run_inference()
