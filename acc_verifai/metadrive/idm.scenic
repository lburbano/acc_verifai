#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from controllers.lateral_control import LateralControl

param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
model scenic.simulators.metadrive.model
param verifaiSamplerType = 'ce' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

#CONSTANTS
TERMINATE_TIME = 20 / globalParameters.time_step

# Parameters of the scenario.
inter_vehivle_disance = Range(30, 60)

# platoon placement 
LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehivle_disance

def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps

## Longitudinal IDM BEHAVIOR
behavior Longitudinal_IDM(id, vehicle_in_front, lane):
	## IDM Parameters (Intelligent Driver Model)
	# Max speed is 22.5 m/s = 80 kmh = 50 mph
	# normal_speed is 19.4 m/s = 70 kmh
	ACC_FACTOR = 1.0
	DEACC_FACTOR = Range(-6,-4)
	target_speed = Range(20, 22.5)
	DISTANCE_WANTED = Range(1.0, 2.0)
	TIME_WANTED = Range(0.1, 1.5)
	delta = 2 # Range(2, 6)      # Acceleration exponent

	lat_control  = LateralControl(globalParameters.time_step)

	while True:
		acceleration = ACC_FACTOR * (1-np.power(max(self.speed, 0) / target_speed, delta))
		gap = (distance from self to vehicle_in_front) - self.length
		d0 = DISTANCE_WANTED
		tau = TIME_WANTED
		ab = -ACC_FACTOR * DEACC_FACTOR
		dv = self.speed - vehicle_in_front.speed
		d_star = d0 + self.speed * tau + vehicle_in_front.speed * dv / (2 * np.sqrt(ab))
		speed_diff = d_star / not_zero(gap)
		acceleration -= ACC_FACTOR * (speed_diff**2)

		if acceleration > 0:
			throttle = min(acceleration, 1)
			brake = 0
		else:
			throttle = 0
			brake = min(-acceleration, 1)

		s = lat_control.compute_control(self, lane)
		take SetThrottleAction(throttle), SetBrakeAction(brake), SetSteerAction(s)

#PLACEMENT
# initLane = network.roads[13].forwardLanes.lanes[0]
# spawnPt = initLane.centerline.pointAlongBy(0)
spawnPt = (-100 @ -48.87)

id = 0
ego = new Car at spawnPt

id = 1
c1 = new Car at ego.position offset by (LEADCAR_TO_EGO, 0),
	with behavior Longitudinal_IDM(id, ego, spawnPt)

id = 2
c2 = new Car at c1.position offset by (C1_TO_C2, 0),
	with behavior Longitudinal_IDM(id, c1, spawnPt)

id = 3
c3 = new Car at c2.position offset by (C2_TO_C3, 0),
	with behavior Longitudinal_IDM(id, c2, spawnPt)


'''
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
'''
# terminate when (simulation().currentTime > TERMINATE_TIME) 
terminate when (distance from ego to c1) < 4.5
terminate when (distance from c1 to c2) < 4.5
terminate when (distance from c2 to c3) < 4.5