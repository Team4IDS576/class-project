from RoadNetEnv import raw_env
from NguyenNetwork import nguyenNetwork

test = nguyenNetwork()

neighbor_nodes = test.neighbors("1")

for i in neighbor_nodes:
    print(type(i))