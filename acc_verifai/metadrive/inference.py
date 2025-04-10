# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# %%
results_plotter.plot_results(
    ["logs/tmp10"],
    num_timesteps=100_000,
    x_axis=results_plotter.X_TIMESTEPS,
    task_name="PPO IDM Attacker",
)

# %%

set_random_seed(0)

scenario = scenic.scenarioFromFile(
        "idm.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(timestep=0.05, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=True, real_time=False),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(4,4)),
    action_space=spaces.Box(low=-1, high=1, shape=(1,)),
    max_steps=600,
)

log_dir="logs/inference"
env = Monitor(env, log_dir) # , info_keywords=("attacker_crashed", "counter_example_found", "dense_reward_signals")


# %%
model = PPO.load("models/ppo_idm_attacker10", env=env)

# %%
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# %%
