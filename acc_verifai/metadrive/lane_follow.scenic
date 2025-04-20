'''
To run this file using the MetaDrive simulator:
    scenic examples/driving/car.scenic --2d --model scenic.simulators.metadrive.model --simulate

To run this file using the Carla simulator:
    scenic examples/driving/car.scenic --2d --model scenic.simulators.carla.model --simulate
'''

param map = localPath('../maps/Town06.xodr')

model scenic.domains.driving.model

spawn_pt = (100 @ -150)
ego = new Car at spawn_pt

targetLane = ego.lane
max_deviation = 1.0
terminate when targetLane.centerline.distanceTo(ego.position) > max_deviation