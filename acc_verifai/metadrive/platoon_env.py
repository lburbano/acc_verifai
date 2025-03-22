from acc_traffic_manager import AccTrafficManager

from metadrive.envs import MetaDriveEnv


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
        r, i = super().reward_function(vehicle_id, *args, **kwargs)

        # Reward function for the platoon attack scenario
        info = {"attack_success": False, "platoon_vehicles": self.platoon_vehicles}
        reward = 0
        ego_vehicle = self.agents[vehicle_id]
        info["platoon_crash"] = self.platoon_vehicles
        # Check for collisions among platoon vehicles (exclude ego)
        for vehicle in self.platoon_vehicles:
            if vehicle.crash_vehicle:
                reward += 1

        # Optionally, penalize any collision involving the ego vehicle
        info["ego_crash"] = ego_vehicle.crash_vehicle
        if ego_vehicle.crash_vehicle:
            reward -= 1

        if reward > 0:
            info["attack_success"] = True

        return reward, info
