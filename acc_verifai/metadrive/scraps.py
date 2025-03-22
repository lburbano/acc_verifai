# %%
from pathlib import Path

from IPython.display import Image

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs import MetaDriveEnv
from metadrive.manager import BaseManager
from metadrive.policy.idm_policy import IDMPolicy


MAX_EPISODE_STEP = 180
TEST_PLATOON = 2


class AccTrafficManager(BaseManager):
    """Manager class for ACC theat model with configurable number of follower vehicles."""

    def __init__(self):
        super().__init__()
        self.platoon_vehicles = []
        self.vehicles = self.engine.global_config["agent_configs"]
        self.num_vehicles = self.engine.global_config["num_vehicles"]

    def before_step(self) -> None:
        """Set actions for all spawned objects before each simulation step."""
        for n, (i, obj) in enumerate(self.spawned_objects.items()):
            p = self.get_policy(i)
            a = p.act()
            if n == 1:
                a[1] = -1.0  # set acceleration to 1.0
            if n == TEST_PLATOON:
                a[1] = 1.0
            obj.before_step(a)  # set action

    def _setup_platoon(self) -> None:
        """Set up the platoon of vehicles for the attack scenario."""
        # Assuming the first vehicle is the ego (attacker)
        self.ego_vehicle = self.vehicles["default_agent"]
        # Create following vehicles with a fixed policy
        for i in range(1, self.num_vehicles):
            # Position vehicles in a one-lane configuration behind the ego
            obj = self.spawn_object(
                DefaultVehicle,
                vehicle_config=dict(
                    spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
                    spawn_longitude=self.ego_vehicle["spawn_longitude"] - (i * 30),  # 30 meters behind
                    spawn_lateral=0,
                ),
            )
            self.platoon_vehicles.append(obj)
            self.add_policy(obj.id, IDMPolicy, obj, self.generate_seed())

    def reset(self) -> None:
        """Reset the traffic manager for a new episode."""
        self._setup_platoon()

    def after_step(self) -> None:
        """Perform any necessary actions after each simulation step."""
        for obj in self.spawned_objects.values():
            obj.after_step()
        if self.episode_step == MAX_EPISODE_STEP:
            self.clear_objects(list(self.spawned_objects.keys()))


# Expand the default config system, specify where to spawn the car
MY_CONFIG = dict(num_vehicles=4)


class PlatoonEnv(MetaDriveEnv):
    """Platoon Environment."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.platoon_vehicles = []

    @classmethod
    def default_config(cls) -> dict:
        """Set default configuration for the environment."""
        config = super().default_config()
        config.update(MY_CONFIG)
        return config

    def setup_engine(self) -> None:
        """Set up the engine with a custom traffic manager."""
        super().setup_engine()
        self.engine.update_manager(
            "traffic_manager",
            AccTrafficManager(),
        )
        self.platoon_vehicles = self.engine.traffic_manager.platoon_vehicles

    def reset(self) -> any:
        """Reset the environment for a new episode."""
        observation = super().reset()
        self.platoon_vehicles = self.engine.traffic_manager.platoon_vehicles
        return observation

    def reward_function(self, vehicle_id: str, *args: dict, **kwargs: dict) -> tuple[float, dict]:
        """Overwrite reward function for the platoon attack scenario."""
        _, i = super().reward_function(vehicle_id, *args, **kwargs)
        i["platoon_crash"] = False
        reward = 0
        ego_vehicle = self.agents[vehicle_id]

        # Check for collisions among platoon vehicles (exclude ego)
        for vehicle in self.platoon_vehicles:
            if vehicle.crash_vehicle:
                reward += 1

        # if attacker crash, negative reward
        if ego_vehicle.crash_vehicle:
            reward -= 1

        return reward, i


map_config = {
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
    BaseMap.GENERATE_CONFIG: "SSSS",
    BaseMap.LANE_WIDTH: 10,
    BaseMap.LANE_NUM: 1,
}

# Example configuration for the environment
config = {
    "num_vehicles": 6,
    "use_render": False,
    "map_config": map_config,
    "agent_configs": {
        "default_agent": dict(
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
            spawn_longitude=150,
            spawn_lateral=0,
        ),
    },
    # Additional configuration parameters for a one-lane road scenario
}

env = PlatoonEnv(config)

# %%
env.reset()
for _ in range(180):
    ob, r, d, t, info = env.step([-0.2, 1.0])
    env.render(
        mode="topdown",
        window=False,
        screen_size=(400, 400),
        camera_position=(120, 7),
        scaling=2,
        screen_record=True,
        text={
            "ego crash": info["crash_vehicle"],
            "platoon crash": info["platoon_crash"],
            "reward": r,
            "Timestep": env.episode_step,
        },
    )
env.top_down_renderer.generate_gif()
env.close()

# %%
Image(Path.open("demo.gif", "rb").read())

# %%
