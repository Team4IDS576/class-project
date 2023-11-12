from pettingzoo.test import parallel_api_test

import old.RoadNetEnv as RoadNetEnv

env = RoadNetEnv.parallel_env()

print("start")

parallel_api_test(env, num_cycles=1000)