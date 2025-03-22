import numpy as np
# import scenic.core.distributions import 
from collections import deque

import scipy.interpolate
from controllers.pid import PID
import copy
import scipy as scipy
class AccControl():
    def __init__(self, id, dt, ego_speed, is_attacker, inter_vehivle_disance, attack_params=None, sampling_attack=5) -> None:
        
        # self.axis = 0

        self.intiliazed = False

        self.vehicle_id = id
        self.dt = dt
        self.d = inter_vehivle_disance
        self.is_attacker = is_attacker
        self.attack_params = attack_params

        if is_attacker:
            self.time = np.linspace(0, sampling_attack*len(attack_params), len(attack_params))
            self.attack_params = attack_params
            self.interpolator = scipy.interpolate.splrep(self.time, self.attack_params)

        self.t = 0

        # Initialize low level controller
        # TODO: move this outside
        self.low_level_control = PID(K_P = 0.06, K_I = 0.35, K_D=0.006, dt=self.dt, min=-1, max=1, ie=5, int_sat=10)
        self.speed_control = PID(K_P=0.4, K_I=0.01, dt = dt, tau=1, int_sat=20, min=-3, max=3)
        self.desired_vel = ego_speed
        # self.low_level_control = PIDLongitudinalController(K_P = 0.18, K_I = 0.08, K_D=0.0005, dt=self.dt, min=-2, max=2)

        self.kp = 0.9304 # These value are gotten from an LQR controller
        self.kd = 2.1599
        

        self.switching_dist = self.d + 25
        self.switching_vel  = 1
        self.eps_dist       = 10    # Avoids zeno
        self.eps_vel        = 1     # Avoids zeno
        self.mode           = 0     # Mode = 0 then speed 
                                    # Mode = 1 then following

    def switch(self, states_leader, states_car):
        dist = states_leader[0] - states_car[0]
        relative_speed = states_leader[1] - states_car[1]
        if self.mode == 0:
            if dist < self.switching_dist and states_leader[1] < self.desired_vel:
                self.mode = 1
        else:
            if dist > self.switching_dist + self.eps_dist or states_leader[1] > self.desired_vel + self.eps_vel:
                self.mode = 0
    
    def follower_control(self, states_leader, states_car):
        
        st = copy.copy(states_car)
        st[0] = states_leader[0] - states_car[0] - self.d
        st[1] = states_leader[1] - states_car[1]
        acceleration_target = self.kp * (st[0]) + self.kd * (st[1])
        return acceleration_target
    
    def cruise_control(self, states_car):
        error = self.desired_vel - states_car[1]
        acceleration_target = self.speed_control.run_step(error)
        return acceleration_target
    
    def acceleration_control(self, acceleration, acceleration_target):
        acc = acceleration
        if acc > 100:
            acc = 0
        acceleration_error = acceleration_target - acc
        action = self.low_level_control.run_step(acceleration_error) 
        return action
    
    def full_control(self, car, leader):
        states_car = np.array([car.position[0], car.velocity[0] ])
        if leader is not None:
            states_leader = np.array([leader.position[0], leader.velocity[0]])

           
            self.switch(states_leader, states_car)
        if self.mode == 0:
            acceleration_target = self.cruise_control(states_car)
        else:
            acceleration_target = self.follower_control(states_leader, states_car)
        
        return acceleration_target

    def compute_control(self, cars):
        '''
        Computes the controller. 
        '''
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
            #action = np.interp(self.t, self.time, self.attack_params)
            action = scipy.interpolate.splev(self.t, self.interpolator).item()
        else:
            acceleration_target = self.full_control(car, leader)
            acceleration = car.carlaActor.get_acceleration().x
            action = self.acceleration_control(acceleration, acceleration_target)

        # print(f'{acceleration_target}, {dist - self.d}, {relative_speed}, {car.carlaActor.get_acceleration().x}')

        
        
        if action > 0:
            throttle = min(1, action )
            brake = 0
        else:
            brake = min(1, abs(action))
            throttle = 0

        return brake, throttle
    

