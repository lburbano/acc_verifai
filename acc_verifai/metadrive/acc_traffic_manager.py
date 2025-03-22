from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.manager import BaseManager
from metadrive.policy.idm_policy import IDMPolicy


MAX_EPISODE_STEP = 180


class AccTrafficManager(BaseManager):
    """Manager class for ACC theat model with configurable number of follower vehicles."""

    def __init__(self):
        super().__init__()
        self.platoon_vehicles = []
        self.vehicles = self.engine.global_config["agent_configs"]
        self.num_vehicles = self.engine.global_config["num_vehicles"]

    def before_step(self) -> None:
        """Set actions for all spawned objects before each simulation step."""
        for i, obj in self.spawned_objects.items():
            p = self.get_policy(i)
            obj.before_step(p.act())  # set action

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
