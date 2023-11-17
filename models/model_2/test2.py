from pettingzoo.test import parallel_api_test

import RoadNetEnv2

env = RoadNetEnv2.parallel_env()

parallel_api_test(env, num_cycles=1000)