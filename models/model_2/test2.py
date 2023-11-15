import RoadNetEnv2

env = RoadNetEnv2.raw_env()

test = env.road_network.neighbors("10")

for item in test:
    print(item)