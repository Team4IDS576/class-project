from RoadNetEnv import raw_env
from NguyenNetwork import nguyenNetwork, traffic

test = raw_env()

print(list(test.road_network.neighbors("8")))
