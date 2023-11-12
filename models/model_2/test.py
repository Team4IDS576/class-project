from pettingzoo.test import parallel_api_test

import RoadNetEnv

env = RoadNetEnv.parallel_env()

print("start")

parallel_api_test(env, num_cycles=1000)