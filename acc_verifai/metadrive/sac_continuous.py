# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as f
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    """Arguments for the experiment."""

    exp_name: str = Path(__file__).name[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    num_envs: int = 1
    """the number of parallel environments"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id: str, seed: int, idx: int, run_name: str, capture_video: int = 0) -> callable:
    """Create a gym environment with optional video recording."""

    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """A neural network model for the Soft Q-learning algorithm."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Soft Q-learning network."""
        x = torch.cat([x, a], 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Actor network for a continuous action space in a reinforcement learning environment."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Actor network."""
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple:
        """Get the action, log probability, and mean for the given input tensor."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """,
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    log_dir = f"../../pvcvolume/runs/{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    model_dir = Path(f"{log_dir}/models/")
    model_dir.mkdir(parents=True, exist_ok=True)
    actor_checkpoint_path = f"{model_dir}/actor.pth"
    qf1_checkpoint_path = f"{model_dir}/qf1.pth"
    qf2_checkpoint_path = f"{model_dir}/qf2.pth"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SubprocVecEnv([make_env(args.env_id, args.seed + i, 0, run_name, args.capture_video) for i in range(args.num_envs)])
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    envs = VecMonitor(envs, log_dir)

    max_action = float(envs.action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(0, args.total_timesteps, args.num_envs):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # record metrics for plotting
        for _, info in enumerate(infos):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                print(f"global_step={global_step}, episodic_return={ep_return}")
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)
                break

        # get real terminal observation
        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                terminal_obs = infos[idx]["terminal_observation"]
                real_next_obs[idx] = terminal_obs
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > (args.learning_starts * args.num_envs):
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = f.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = f.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % (args.policy_frequency * args.num_envs) == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency * args.num_envs,
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % (args.target_network_frequency * args.num_envs) == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters(), strict=False):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters(), strict=False):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    print(f"Saving model checkpoint at step {global_step} to {actor_checkpoint_path}")
    torch.save(actor.state_dict(), actor_checkpoint_path)
    torch.save(qf1.state_dict(), qf1_checkpoint_path)
    torch.save(qf2.state_dict(), qf2_checkpoint_path)
