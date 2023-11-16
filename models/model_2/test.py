from pettingzoo.test import parallel_api_test

import RoadNetEnv2 as rne

env = rne.parallel_env()

print("start")

parallel_api_test(env, num_cycles=10000)