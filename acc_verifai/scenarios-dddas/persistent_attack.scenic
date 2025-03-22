from controllers.lateral_control   import  LateralControl
from controllers.acc               import  AccControl
 


param map                      = localPath('../maps/Town06.xodr')
param carla_map                = 'Town06'
param time_step                = 1.0/10
param verifaiSamplerType       = 'bo'

model scenic.simulators.carla.model


# Parameters of the scenario.
EGO_SPEED                   = 20

#CONSTANTS
TERMINATE_TIME = 25 / globalParameters.time_step   # 250 time steps
MODEL          = "vehicle.tesla.model3"


############
# Attack params
# TODO: tune these parameters
############

param   amplitude_acc   =  VerifaiRange (0,    1)
param   frequency 		=  VerifaiRange (0,   10)
param   attack_time 	=  VerifaiRange (0,   10)


############
# vehicle distances are measured from the center of mass
# so in reality the bumper to bumper is var - 4.95
############
inter_vehicle_distance = 7
LEADCAR_TO_EGO = C1_TO_C2 = C2_TO_C3 = -inter_vehicle_distance

## DEFINING BEHAVIORS

#EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior Attacker(id, dt, ego_speed, lane):
	attack_params = { 'amplitude_acc': globalParameters.amplitude_acc,
					  'frequency': globalParameters.frequency,
					  'attack_time': globalParameters.attack_time }

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
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
terminate when simulation().currentTime > TERMINATE_TIME
'''
terminate when (distance from ego to start) > 760

terminate when simulation().currentTime > TERMINATE_TIME