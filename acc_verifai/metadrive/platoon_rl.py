# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


# %%


def make_env() -> callable:
    """Create a function that returns a new environment instance."""

    def thunk() -> ScenicGymEnv:
        scenario = scenic.scenarioFromFile(
            "baseline-av.scenic",
            model="scenic.simulators.metadrive.model",
            mode2D=True,
        )

        env = ScenicGymEnv(
            scenario,
            MetaDriveSimulator(timestep=0.02, sumo_map=pathlib.Path("../maps/Town06.net.xml"), render=False, real_time=False),
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(258,)),
            action_space=spaces.Box(low=-1, high=1, shape=(2,)),
            max_steps=700,
        )
        return env

    return thunk


# %%
def main() -> None:
    """Run RL training."""
    set_random_seed(0)

    log_dir = "../../../../pvcvolume/generic-av-baseline"

    envs = [make_env() for i in range(8)]
    env = SubprocVecEnv(envs)
    env = VecMonitor(env, log_dir)
    env.reset()

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000, log_interval=1, progress_bar=True)
    model.save("generic-av-baseline")

    env.close()


if __name__ == "__main__":
    main()
