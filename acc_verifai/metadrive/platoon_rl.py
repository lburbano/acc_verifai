# %%
import pathlib

import numpy as np
import scenic
from gymnasium import spaces
from scenic.gym import ScenicGymEnv
from scenic.simulators.metadrive import MetaDriveSimulator


# %%

scenario = scenic.scenarioFromFile(
    "idm.scenic",
    model="scenic.simulators.metadrive.model",
    mode2D=True,
)

# %%

env = ScenicGymEnv(
    scenario,
    MetaDriveSimulator(sumo_map=pathlib.Path("../maps/Town06.net.xml")),
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(4,5)),
    action_space=spaces.Box(low=-1, high=1, shape=(2,)),
)

# %%
env.reset()

# %%
while True:
    obs, reward, terminated, truncated, info = env.step(np.array([1,1]))
    print(f"reward: {reward}")
    if terminated or truncated:
        env.reset()

# %%
