import copy

import numpy as np

# import scenic.core.distributions import
from controllers.pid import PID


ACC = 100


class AccControl:
    """Longitudinal control using a PID to reach a target speed."""
    def __init__(self, vehicle_id: int, dt: float, ego_speed: float, is_attacker: int, attack_params: dict | None = None) -> None:

        self.intiliazed = False

        self.vehicle_id = vehicle_id
        self.dt = dt
        self.d = 7 #control distance
        self.is_attacker = is_attacker
        self.attack_params = attack_params
        self.t = 0

        # Initialize low level controller
        # TODO: move this outside
        self.low_level_control = PID(K_P = 0.1, K_I = 0.1, K_D=0.005, dt=self.dt, min=-1, max=1, ie=5, int_sat=10)
        self.speed_control = PID(K_P=0.4, K_I=0.01, dt = dt, tau=1, int_sat=20, min=-3, max=3)
        self.desired_vel = ego_speed

        self.kp = 0.9304 # These value are gotten from an LQR controller
        self.kd = 2.1599

        self.switching_dist = self.d + 25
        self.switching_vel  = 1
        self.eps_dist       = 10    # Avoids zeno
        self.eps_vel        = 1     # Avoids zeno
        self.mode           = 0     # Mode = 0 then speed
                                    # Mode = 1 then following
        if self.attack_params is not None and "attack_times" in self.attack_params:#["attack_times"]:
            np_array = np.array(self.attack_params["attack_times"])
            attack_times = np.round(np_array).astype(int)
            self.attack_range1 = (attack_times[0], attack_times[1])
            self.attack_range2 = (attack_times[2], attack_times[3])
            self.intensity1 = self.attack_params["intensities"][0]
            self.intensity2 = self.attack_params["intensities"][1]

    def switch(self, states_leader: np.ndarray, states_car: np.ndarray) -> None:
        """Switch between speed control and following control based on the distance between the leader and the car."""
        dist = states_leader[0] - states_car[0]
        states_leader[1] - states_car[1]
        if self.mode == 0:
            if dist < self.switching_dist and states_leader[1] < self.desired_vel:
                self.mode = 1
        elif dist > self.switching_dist + self.eps_dist or states_leader[1] > self.desired_vel + self.eps_vel:
            self.mode = 0

    def follower_control(self, states_leader: np.ndarray, states_car: np.ndarray) -> float:
        """Control the follower car to follow the leader car."""
        st = copy.copy(states_car)
        st[0] = states_leader[0] - states_car[0] - self.d
        st[1] = states_leader[1] - states_car[1]
        acceleration_target = self.kp * (st[0]) + self.kd * (st[1])
        return acceleration_target

    def cruise_control(self, states_car: np.ndarray) -> float:
        """Control the car to cruise at a desired speed."""
        error = self.desired_vel - states_car[1]
        acceleration_target = self.speed_control.run_step(error)
        return acceleration_target

    def acceleration_control(self, acceleration: np.ndarray, acceleration_target: np.ndarray) -> float:
        """Control the acceleration of the car."""
        acc = acceleration
        if acc > ACC:
            acc = 0
        acceleration_error = acceleration_target - acc
        action = self.low_level_control.run_step(acceleration_error)
        return action

    def full_control(self, car: any, leader: any) -> float:
        """Control the car based on the mode."""
        states_car = np.array([car.position[0], car.velocity[0] ])
        if leader is not None:
            states_leader = np.array([leader.position[0], leader.velocity[0]])
            self.switch(states_leader, states_car)
        if self.mode == 0:
            acceleration_target = self.cruise_control(states_car)
        else:
            acceleration_target = self.follower_control(states_leader, states_car)
        return acceleration_target

    def compute_control(self, cars: any) -> tuple[float, float]:  # noqa: PLR0912
        """Compute the controller."""
        if not self.intiliazed:
            self.intiliazed = True
            return 0, 0

        self.t += self.dt
        car = cars[self.vehicle_id]
        if self.vehicle_id > 0:
            leader = cars[self.vehicle_id - 1]
        else:
            leader = None

        if self.is_attacker:
            if "attack_times" in self.attack_params:
                print(self.attack_params["attack_times"])
                np_array = np.array(self.attack_params["attack_times"])
                self.attack_times = np.round(np_array).astype(int)
                r1, r2 = self.attack_range1
                r3, r4 = self.attack_range2

                print(self.attack_times)
                print("intensities")
                print(self.attack_params["intensities"])
                if self.t >= r1 and self.t <= r2:
                    action = self.intensity1
                elif self.t >= r3 and self.t <= r4:
                    action = self.intensity2
                else:
                    acceleration_target = self.full_control(car, leader)
                    acceleration = car.carlaActor.get_acceleration().x
                    action = self.acceleration_control(acceleration, acceleration_target)
            elif self.attack_params["attack_time"]:
                if self.t < self.attack_params["attack_time"]:
                    acceleration_target = self.full_control(car, leader)
                    acceleration = car.x
                    action = self.acceleration_control(acceleration, acceleration_target)
                else:
                    action = np.sign( np.sin(self.t * self.attack_params["frequency"]))
                    action *= self.attack_params["amplitude_acc"]
            else:
                acceleration_target = self.full_control(car, leader)
                acceleration = car.x
                action = self.acceleration_control(acceleration, acceleration_target)
        else:
            acceleration_target = self.full_control(car, leader)
            acceleration = car.x
            print(f"acceleration: {acceleration}", f"acceleration_target: {acceleration_target}")
            action = self.acceleration_control(acceleration, acceleration_target)

        if action > 0:
            throttle = min(1, action )
            brake = 0
        else:
            brake = min(1, abs(action))
            throttle = 0

        return brake, throttle
