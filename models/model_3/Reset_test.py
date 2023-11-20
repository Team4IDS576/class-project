from pettingzoo.test import parallel_api_test

import RoadNetEnv3_debugging as rne

env = rne.parallel_env()

state, _ = env.reset()

test_action = {agent_id: 0 for agent_id in env.agents}

# print("State before reset")
print(env.state())

for _ in range(100):
    env.step(test_action)
    print(env.state())

print("\n\n\n\n")
# print("State after step")
#print(env.state())

state, _ = env.reset()

# print("State after reset")
print(env.state())