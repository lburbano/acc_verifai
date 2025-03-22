import numpy as np
# import scenic.core.distributions import 
from controllers.pid import PID

class LateralControl():
    def __init__(self, dt) -> None:
        self.pid = PID(K_P=0.1, K_D=0.1, K_I=0.01, dt=dt, tau=0.1)
        self.K_P = 0.1
        self.K_D = 0.1
        self.K_I = 0.01
        self.current_steering = 0
        
    def compute_control(self, car, lane_center):
        position = car.position
        position = position[1]
        error = lane_center[1] - position
        speed = abs(car.speed)
        if speed > 0.1:
            self.current_steering = self.pid.run_step(-error)

        if speed < 10:
            div = 1
        elif speed < 20:
            div = 2
        elif speed < 30:
            div = 4
        else:
            div = 8
        # if steering - self.current_steering > 0.1:
        #     steering = self.current_steering + 0.1
        # elif steering - self.current_steering < 0.1:
        #     steering = self.current_steering - 0.1
        # self.current_steering = np.clip(steering, -1, 1)
        
        
        return self.current_steering/div