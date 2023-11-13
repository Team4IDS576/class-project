from ray.rllib.utils import check_env

import RoadNetEnvEncoded as rne
#from pettingzoo.butterfly import pistonball_v6

env = rne.parallel_env()

test = env.step(0)

print(test)