#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
import math
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy

param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
# model scenic.simulators.lgsvl.model
model scenic.simulators.carla.model
# param render = True
param verifaiSamplerType = 'ce' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

# Parameters of the scenario.
EGO_SPEED = 20
param EGO_BRAKING_THRESHOLD = VerifaiRange(5, 15)

#CONSTANTS
TERMINATE_TIME = 40 / globalParameters.time_step
CAR3_SPEED = 20
CAR4_SPEED = 20
LEAD_CAR_SPEED = 20
MODEL = "vehicle.tesla.model3"


############
# Attack params
# TODO: tune these parameters
############
amplitude_brake = VerifaiRange(0, 1)
amplitude_acc   = VerifaiRange(0, 1)
frequency 		= VerifaiRange(0, 10)
attack_time 	= VerifaiRange(0, 10)
duty_cycle      = VerifaiRange(0, 1)


############

inter_vehivle_disance = 60

LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehivle_disance

DT = 5
C3_BRAKING_THRESHOLD = 6
C4_BRAKING_THRESHOLD = 6
LEADCAR_BRAKING_THRESHOLD = 6


## DEFINING BEHAVIORS
#COLLISION AVOIDANCE BEHAVIOR
behavior CollisionAvoidance(safety_distance=10):
	take SetBrakeAction(BRAKE_ACTION)

#EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior Attacker(id, dt, ego_speed, lane):
	# TODO: implement with RL agent actions
	take SetThrottleAction(0), SetBrakeAction(0)

# CAR4 BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior Follower(id, vehicle_in_front):
	a = 23.0      # Maximum acceleration
	b = 1.6       # Comfortable deceleration
	v0 = 23.0     # Desired acceleration
	s0 = 5.0      # Minimum gap
	T = 1.5       # Safe time headway (s)
	delta = 4  # Acceleration exponent
	dt = 0.1

	while True:
		# acceleration
		gap = (distance from self to vehicle_in_front) - self.length
		delta_v = self.velocity[0] - vehicle_in_front.velocity[0]
		s_star = s0 + self.velocity[0] * T + (self.velocity[0] * delta_v) / (2 * math.sqrt(a * b))
		acceleration = a * (1 - ((self.velocity[0]/v0)**delta) - (s_star / gap)**2)

		if acceleration > 0:
			throttle = min(acceleration, 1)
			brake = 0
		else:
			throttle = 0
			brake = min(-acceleration, 1)
		print(f"throttle: {throttle}, brake: {brake}")
		take SetThrottleAction(throttle), SetBrakeAction(brake), SetSteerAction(0)

#PLACEMENT
# initLane = network.roads[0].forwardLanes.lanes[0]
# spawnPt = initLane.centerline.pointAlongBy(SPAWN)
start = (-100 @ -48.87)

id = 0
ego = new Car at start,
    with behavior Attacker(id, globalParameters.time_step, EGO_SPEED-5, start),

id = 1
c1 = new Car at ego.position offset by (LEADCAR_TO_EGO, 0),
	with behavior Follower(id, ego)

id = 2
c2 = new Car at c1.position offset by (C1_TO_C2, 0),
	with behavior Follower(id, c1)

id = 3
c3 = new Car at c2.position offset by (C2_TO_C3, 0),
	with behavior Follower(id, c2)


'''
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
terminate when simulation().currentTime > TERMINATE_TIME
'''
terminate when (distance from ego to start) > 760