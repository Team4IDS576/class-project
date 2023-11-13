import RoadNetEnv
import ray

env = RoadNetEnv.parallel_env()

ray.rllib.utils.check_env(env)