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
param verifaiSamplerType = 'halton' # TODO: use scenic/random/uniform/halton sampler to train from scratch; then use ce for fine-tuning

#CONSTANTS
TERMINATE_TIME = 40 / globalParameters.time_step

def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps

def get_vehicle_ahead(id, vehicle, lane):
	""" Returns the closest object in front of the vehicle that is:
	(1) visible,
	(2) on the same lane (or intersection),
	within the thresholdDistance.
	Returns the object if found, or None otherwise. """
	closest = None
	minDistance = float('inf')
	objects = simulation().objects
	for obj in objects:
		if not (vehicle can see obj):
			continue
		d = abs(vehicle.position.x - obj.position.x) # (distance from vehicle.position to obj.position)
		if vehicle == obj or d < 0.1:
			continue
		if lane != obj._lane:
			continue
		if d < minDistance:
			minDistance = d
			closest = obj
	return closest

def get_vehicle_behind(id, vehicle, lane):
	""" Returns the closest object behind the vehicle that is:
	(1) visible,
	(2) on the same lane (or intersection),
	within the thresholdDistance.
	Returns the object if found, or None otherwise. """
	closest = None
	minDistance = float('inf')
	objects = simulation().objects
	for obj in objects:
		if not (obj can see vehicle):
			continue
		d = abs(obj.position.x - vehicle.position.x)
		if vehicle == obj or d < 0.1:
			continue
		if lane != obj._lane:
			continue
		if d < minDistance:
			minDistance = d
			closest = obj
	return closest

def get_adjacent_lane(id, vehicle, direction):
	"""Get the adjacent lane in the specified direction ('left' or 'right') from the current lane."""
	lane_section = network.laneSectionAt(vehicle.position)
	if lane_section is None:
		return None

	if direction == "left" and lane_section._laneToLeft:
		return lane_section.laneToLeft.lane
	if direction == "right" and lane_section._laneToRight:
		return lane_section.laneToRight.lane

	return None

def map_acc_to_throttle_brake(acc, max_throttle=1, max_brake=1):
	if acc > 0:
		throttle = min(acc, max_throttle)
		brake = 0
	else:
		throttle = 0
		brake = min(abs(acc), max_brake)
	return throttle, brake

def regulateSteering(steer, past_steer, max_steer=0.8):
	# Steering regulation: changes cannot happen abruptly, can't steer too much.
	if steer > past_steer + 0.1:
		steer = past_steer + 0.1
	elif steer < past_steer - 0.1:
		steer = past_steer - 0.1
	if steer >= 0:
		steer = min(max_steer, steer)
	else:
		steer = max(-max_steer, steer)
	return steer

def idm_acc(agent, vehicle_in_front, acc_factor=1.0, deacc_factor=-2, target_speed=10, distance_wanted=2, time_wanted=1.5, delta=2):
	if not agent:
		return 0.0

	acceleration = acc_factor * (1-np.power(max(agent.speed, 0) / target_speed, delta))
	if vehicle_in_front is None:
		return acceleration

	gap = (vehicle_in_front.position.x - agent.position.x) - agent.length
	d0 = distance_wanted
	tau = time_wanted
	ab = -acc_factor * deacc_factor
	dv = agent.speed - vehicle_in_front.speed
	d_star = d0 + agent.speed * tau + vehicle_in_front.speed * dv / (2 * np.sqrt(ab))
	speed_diff = d_star / not_zero(gap)
	acceleration -= acc_factor * (speed_diff**2)
	return acceleration

behavior IDM_MOBIL(id, politeness=0.25):
	# IDM params
	acc_factor = 1.0
	deacc_factor = Range(-3,-1)
	target_speed = Range(10, 15)
	distance_wanted = Range(1.0, 2.0)
	time_wanted = Range(0.1, 1.5)
	delta = Range(2, 6)
	lane_change_min_acc_gain = 1.0
	safe_braking_limit = 1.0

	_lon_controller_follow, _lat_controller_follow = simulation().getLaneFollowingControllers(self)
	_lon_controller_change, _lat_controller_change = simulation().getLaneChangingControllers(self)
	past_steer_angle = 0
	current_lane = self._lane
	current_centerline = current_lane.centerline

	while True:
		current_lane = network.laneAt(self.position)
		if current_lane is None:
			break
		current_centerline = current_lane.centerline

		vehicle_front = get_vehicle_ahead(id, self, current_lane)

		# Lateral: MOBIL
		best_change_advantage = -float('inf')
		target_lane_for_change = None
		if vehicle_front:
			for direction in ["left", "right"]:
				adjacent_lane = get_adjacent_lane(id, self, direction)
				if adjacent_lane is None or adjacent_lane == current_lane:
					continue

				# find relevant vehicles for MOBIL calculation
				ego_leader = get_vehicle_ahead(id, self, current_lane)
				ego_follower = get_vehicle_behind(id, self, current_lane)
				adjacent_leader = get_vehicle_ahead(id, self, adjacent_lane)
				adjacent_follower = get_vehicle_behind(id, self, adjacent_lane)

				# Is the maneuver unsafe for the new following vehicle?
				adjacent_follower_acc = idm_acc(adjacent_follower, adjacent_leader, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
				adjacent_follower_pred_acc = idm_acc(adjacent_follower, self, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
				if adjacent_follower_pred_acc < -safe_braking_limit:
					continue
				
				# Is there an acceleration advantage for me and/or my followers to change lane?
				ego_pred_acc = idm_acc(self, adjacent_leader, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
				ego_acc = idm_acc(self, ego_leader, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
				ego_follower_acc = idm_acc(ego_follower, self, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
				ego_follower_pred_acc = idm_acc(ego_follower, ego_leader, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)

				incentive = (ego_pred_acc - ego_acc) + politeness * ((adjacent_follower_pred_acc - adjacent_follower_acc) + (ego_follower_pred_acc - ego_follower_acc))
				if incentive < lane_change_min_acc_gain:
					continue
				target_lane_for_change = adjacent_lane

		if target_lane_for_change:
			change_centerline = target_lane_for_change.centerline
			while abs(change_centerline.signedDistanceTo(self.position)) > 0.3:
				# Lateral: Lane change
				cte = change_centerline.signedDistanceTo(self.position)
				current_steer_angle = _lat_controller_change.run_step(cte)
				current_steer_angle = regulateSteering(current_steer_angle, past_steer_angle)
				
				# Longitudinal: throttle/brake
				leader_during_change = get_vehicle_ahead(id, self, target_lane_for_change)
				acceleration = idm_acc(self, leader_during_change, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
				throttle, brake = map_acc_to_throttle_brake(acceleration)

				take SetThrottleAction(throttle), SetBrakeAction(brake), SetSteerAction(current_steer_angle)
				past_steer_angle = current_steer_angle
			current_lane = target_lane_for_change
			current_centerline = current_lane.centerline
		else:
			acceleration = idm_acc(self, vehicle_front, target_speed=target_speed, distance_wanted=distance_wanted, time_wanted=time_wanted, delta=delta)
			throttle, brake = map_acc_to_throttle_brake(acceleration)

			nearest_line_points = current_centerline.nearestSegmentTo(self.position)
			nearest_line_segment = PolylineRegion(nearest_line_points)
			cte = nearest_line_segment.signedDistanceTo(self.position)
			current_steer_angle = _lat_controller_follow.run_step(cte) # Use the lane following lateral controller
			current_steer_angle = regulateSteering(current_steer_angle, past_steer_angle)

			take SetThrottleAction(throttle), SetBrakeAction(brake), SetSteerAction(current_steer_angle)
			past_steer_angle = current_steer_angle

behavior dummy_attacker():
	while True:
		take SetThrottleAction(1.0), SetBrakeAction(0.0), SetSteerAction(0.0)

#PLACEMENT
ego_spawn_pt  = (100 @ -146.5)

num_vehicles_to_place = 6
lane_width = 3.5

id = 0
ego = new Car on ego_spawn_pt

victim_spawn_pt = (120 @ -150)

victim_vehicles = []
for i in range(num_vehicles_to_place):
	victim_longitudinal_offset = VerifaiRange(0, 200)
	c_i_spawn_pt = (victim_spawn_pt[0] + victim_longitudinal_offset) @ victim_spawn_pt[1]

	lane_group = network.laneGroupAt(c_i_spawn_pt)
	lane_i = Uniform(*lane_group.lanes)
	c_i = new Car on lane_i, with behavior IDM_MOBIL(id, politeness=VerifaiRange(0,0.5)) 
	victim_vehicles.append(c_i)

'''
require always (distance from ego.position to c1.position) > 4.99
terminate when ego.lane == None 
'''
terminate when ego.position.x >= ego_spawn_pt[0] + 350
terminate when ego.metaDriveActor.crash_vehicle
terminate when ego._lane == None
terminate when any(v.metaDriveActor.crash_vehicle for v in victim_vehicles)


# terminate when (distance from c1 to c2) < 4.5
# terminate when (distance from c2 to c3) < 4.5
