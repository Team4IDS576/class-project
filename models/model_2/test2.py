from pettingzoo.test import api_test

import RoadNetEnv2

env = RoadNetEnv2.raw_env()

api_test(env, num_cycles=100)