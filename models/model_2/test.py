from pettingzoo.test import parallel_api_test

import RoadNetEnv2 as rne

env = rne.parallel_env()


state, _ = env.reset()

test_action = {agent_id: 0 for agent_id in env.agents}

print(env.state())

for _ in range(10):
    env.step(test_action)

print("\n\n\n\n")
state, _ = env.reset()

print(env.state())