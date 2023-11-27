import RoadNetEnv

env = RoadNetEnv.raw_env()
env.reset()

print(env.agents)