from pettingzoo.test import parallel_api_test, api_test

# import RoadNetEnv3
import RoadNetEnv3_latency_test

# env = RoadNetEnv3.parallel_env()
env = RoadNetEnv3_latency_test.parallel_env()

parallel_api_test(env, num_cycles=1000)

# env = RoadNetEnv3.raw_env()
env = RoadNetEnv3_latency_test.parallel_env()

api_test(env, num_cycles=1000)

