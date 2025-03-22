#SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
# Imports
from controllers.acc import AccControl
from controllers.lateral_control import LateralControl


param map = localPath('../maps/Town06.xodr')
param carla_map = 'Town06'
param time_step = 1.0/10
# model scenic.simulators.lgsvl.model
model scenic.simulators.carla.model
# param render = True
param verifaiSamplerType = 'ce'
param 'sim/weather/cloud_type[0]' = 0
param 'sim/weather/rain_percent' = 0
param weather = 'ClearSunset'

# Parameters of the scenario.
EGO_SPEED = 25
param EGO_BRAKING_THRESHOLD = VerifaiRange(5, 15)

#CONSTANTS
TERMINATE_TIME = 40 / globalParameters.time_step
CAR3_SPEED = 20
CAR4_SPEED = 20
LEAD_CAR_SPEED = 20
MODEL = "vehicle.tesla.model3"
CRASH_DISTANCE = 4.95

############
# Attack params
# TODO: tune these parameters
############
param attack_0 = VerifaiRange(-1, 1) # 0-5
param attack_5 = VerifaiRange(-1, 1) # 5-10
param attack_10 = VerifaiRange(-1, 1) # 10-15
param attack_15 = VerifaiRange(-1, 1) # 15-20
param attack_20 = VerifaiRange(-1, 1) # 20-25
param attack_25 = VerifaiRange(-1, 1) # 25-30
param attack_30 = VerifaiRange(-1, 1) # 30-35
param attack_35 = VerifaiRange(-1, 1) # 35-40

DT = 5


############
# distance between center of mass so in reality the bumper to bumper is 
# var - 4.95
inter_vehivle_disance = 13

LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehivle_disance

C3_BRAKING_THRESHOLD = 6
C4_BRAKING_THRESHOLD = 6
LEADCAR_BRAKING_THRESHOLD = 6


## DEFINING BEHAVIORS
#COLLISION AVOIDANCE BEHAVIOR
behavior CollisionAvoidance(safety_distance=10):
	take SetBrakeAction(BRAKE_ACTION)


#EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior Attacker(id, dt, ego_speed, lane):
	
	attack = [globalParameters.attack_0, globalParameters.attack_5, globalParameters.attack_10, globalParameters.attack_15, globalParameters.attack_20, globalParameters.attack_25, globalParameters.attack_30, globalParameters.attack_35]

	long_control = AccControl(id, dt, ego_speed, True, inter_vehivle_disance, attack, DT)
	lat_control  = LateralControl(globalParameters.time_step)
	while True:
		cars = [ego, c1, c2, c3]
		b, t = long_control.compute_control(cars)
		s = lat_control.compute_control(self, lane)
		take SetThrottleAction(t), SetBrakeAction(b), SetSteerAction(s)

#CAR4 BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior Follower(id, dt, ego_speed, lane):
	long_control = AccControl(id, dt, ego_speed, False, inter_vehivle_disance)
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
	with blueprint MODEL,
	with color Color(1,1,1)


id = 1
c1 = Car at ego.position offset by (LEADCAR_TO_EGO, 0),
	with blueprint MODEL,
	with behavior Follower(id, globalParameters.time_step, EGO_SPEED, start),
	with color Color(1,1,1)


id = 2
c2 = Car at c1.position offset by (C1_TO_C2, 0),
	with blueprint MODEL,
	with behavior Follower(id, globalParameters.time_step, EGO_SPEED, start),
	with color Color(1,1,1)



id = 3
c3 = Car at c2.position offset by (C2_TO_C3, 0),
	with blueprint MODEL,
	with behavior Follower(id, globalParameters.time_step, EGO_SPEED, start),
	with color Color(1,1,1)


'''
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
terminate when simulation().currentTime > TERMINATE_TIME
'''

terminate when (withinDistanceToAnyObjs(ego, CRASH_DISTANCE))
terminate when (withinDistanceToAnyObjs(c1, CRASH_DISTANCE))
terminate when (withinDistanceToAnyObjs(c2, CRASH_DISTANCE))
terminate when (withinDistanceToAnyObjs(c3, CRASH_DISTANCE))

terminate when (distance from ego to start) > 760