from pettingzoo.test import parallel_api_test

# import RoadNetEnv3_debugging as rne
import RoadNetEnv3_latency_test as rne

env = rne.parallel_env()

state, _ = env.reset()
print("State before reset")
print(env.state())

test_action = {agent_id: 0 for agent_id in env.agents}
print("State after step")
for _ in range(10):
    env.step(test_action)    
    print(env.state())

# print("\n\n\n\n")

state, _ = env.reset()
print("State after reset")
print(env.state())