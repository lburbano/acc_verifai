# %%
import pathlib

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import scenic
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator


# %%
def main() -> None:
    """Run RL training."""
    set_random_seed(0)

    scenario = scenic.scenarioFromFile(
        "idm.scenic",
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

    log_dir="logs/tmp6/"

    env = ScenicGymEnv(
        scenario,
        MetaDriveSimulator(timestep=0.05, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=True, real_time=False),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(7,5)),
        action_space=spaces.Box(low=-1, high=1, shape=(2,)),
        max_steps=700,
    )
    env = Monitor(env, log_dir)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save("models/ppo_idm_attacker6")

if __name__== "__main__":
    main()
