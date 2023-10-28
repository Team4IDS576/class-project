import random

import os
import sys

from test_net import test_net
from environment import RoadNetworkEnv

if __name__ == "__main__":
    net = test_net()
    
    env = RoadNetworkEnv(graph = net)
    
    num_episodes = 5
    max_steps_per_episode = 10
    
    for episode in range(num_episodes):
        obs = env.reset()

        for step in range(max_steps_per_episode):
            actions = {}
            agents_to_remove = {}

            for agent in obs:
                if not obs[agent]['neighbors']:
                    agents_to_remove[agent] = "No neighbors"
                    continue

                action = random.choice(obs[agent]['neighbors'])
                next_obs, reward, done, _ = env.step(action, agent)
                obs[agent] = next_obs

                if done:
                    print(f"Agent {agent} reached the destination in {step + 1} steps.")
                    agents_to_remove[agent] = "Reached destination"
                else:
                    print(f"Agent {agent} received reward: {reward}")
            
            for agent, reason in agents_to_remove.items():
                print(f"Agent {agent} removed: {reason}")
                obs.pop(agent, None)

            if len(obs) == 0:
                break
