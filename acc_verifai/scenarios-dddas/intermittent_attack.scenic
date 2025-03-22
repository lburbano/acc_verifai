#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
from controllers.acc import AccControl
from controllers.lateral_control import LateralControl


param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
model scenic.simulatmors.carla.model
# define the sampler type
param verifaiSamplerType = 'ce'

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
param random_t1 = VerifaiRange(10, 20)
param random_t2 = VerifaiRange(21, 30)
param random_t3 = VerifaiRange(31, 40)
param random_t4 = VerifaiRange(41, 50)
param random_thr_br1 = VerifaiRange(0, 1)
param random_thr_br2 = VerifaiRange(0, 1)

############
inter_vehicle_distance = 7
LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehicle_distance

C3_BRAKING_THRESHOLD = 6
C4_BRAKING_THRESHOLD = 6
LEADCAR_BRAKING_THRESHOLD = 6


## DEFINING BEHAVIORS
#COLLISION AVOIDANCE BEHAVIOR
behavior CollisionAvoidance(safety_distance=10):
	take SetBrakeAction(BRAKE_ACTION)


#EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior Attacker(id, dt, ego_speed, lane):
	attack_params = {	'attack_times': [globalParameters.random_t1,globalParameters.random_t2,globalParameters.random_t3,globalParameters.random_t4],
						'intensities': [globalParameters.random_thr_br1,globalParameters.random_thr_br2]
						}

	long_control = AccControl(id, dt, ego_speed, True, inter_vehicle_distance, attack_params)
	lat_control  = LateralControl(globalParameters.time_step)
	while True:
		cars = [ego, c1, c2, c3]
		b, t = long_control.compute_control(cars)
		s = lat_control.compute_control(self, lane)
		take SetThrottleAction(t), SetBrakeAction(b), SetSteerAction(s)

#CAR4 BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior Follower(id, dt, ego_speed, lane):
	long_control = AccControl(id, dt, ego_speed, False, inter_vehicle_distance)
	lat_control  = LateralControl(globalParameters.time_step)
	while True:
		cars = [ego, c1, c2, c3]
		b, t = long_control.compute_control(cars)
		s = lat_control.compute_control(self, lane)
		take SetThrottleAction(t), SetBrakeAction(b), SetSteerAction(s)

#PLACEMENT
# initLane = network.roads[0].forwardLanes.lanes[0]
# spawnPt = initLane.centerline.pointAlongBy(SPAWN)
start = (-100 @ -48.87)

id = 0
ego = Car at start,
    with behavior Attacker(id, globalParameters.time_step, EGO_SPEED-5, start),
	with blueprint MODEL


id = 1
c1 = Car at ego.position offset by (LEADCAR_TO_EGO, 0),
	with blueprint MODEL,
	with behavior Follower(id, globalParameters.time_step, EGO_SPEED, start)


id = 2
c2 = Car at c1.position offset by (C1_TO_C2, 0),
		with blueprint MODEL,
		with behavior Follower(id, globalParameters.time_step, EGO_SPEED, start)



id = 3
c3 = Car at c2.position offset by (C2_TO_C3, 0),
		with blueprint MODEL,
		with behavior Follower(id, globalParameters.time_step, EGO_SPEED, start)
		

'''
require always (distance from ego.position to c1.position) > 4.70
terminate when ego.lane == None 
terminate when simulation().currentTime > TERMINATE_TIME
'''
#require always (distance from ego.position to c1.position) > 4.99
terminate when (distance from ego to start) > 760