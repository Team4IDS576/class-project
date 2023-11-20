import RoadNetEnv3
import pandas as pd
import torch
import numpy as np
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.algorithms.maddpg import MADDPG
from tqdm import trange

# instantiate env and torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = RoadNetEnv3.parallel_env()
env.reset()

# configure algo input parameters
try:
    state_dim = [env.observation_space(agent).n for agent in env.agents]
    one_hot = True
except Exception:
    state_dim = [env.observation_space(agent).shape for agent in env.agents]
    one_hot = False
try:
    action_dim = [env.action_space(agent).n for agent in env.agents]
    discrete_actions = True
    max_action = None
    min_action = None
except Exception:
    action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
    discrete_actions = False
    max_action = [env.action_space(agent).high for agent in env.agents]
    min_action = [env.action_space(agent).low for agent in env.agents]

n_agents = env.num_agents
agent_ids = [agent_id for agent_id in env.agents]
done = {agent_id: False for agent_id in env.agents}
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    memory_size=1_000_000,
    field_names=field_names,
    agent_ids=agent_ids,
    device=device
)

NET_CONFIG = {
    "arch": "mlp",
    "h_size": [64, 64]
}

agent = MADDPG(
    state_dims=state_dim,
    action_dims=action_dim,
    one_hot=one_hot,
    n_agents=n_agents,
    agent_ids=agent_ids,
    max_action=max_action,
    min_action=min_action,
    discrete_actions=True,
    device=device,
    net_config=NET_CONFIG,
)

episodes = 5
max_steps = 100
epsilon = 1.0
eps_end = 0.1
eps_decay = 0.995

episode_travel_times = []
travel_times_df = pd.DataFrame(columns=agent_ids)

for ep in trange(episodes):
    state, info = env.reset()
    agent_reward = {agent_id: 0 for agent_id in env.agents}
    
    for i in range(max_steps):
        agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
        env_defined_actions = (
            info["env_defined_actions"]
            if "env_defined_actions" in info.keys()
            else None
        )
        
        # get next action from agent
        cont_actions, discrete_action = agent.getAction(
            state, epsilon, agent_mask, env_defined_actions
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions
        
        # act in environment
        next_state, reward, termination, truncation, info = env.step(
            action
        )
        
        # save experience to replay buffer
        memory.save2memory(state, cont_actions, reward, next_state, done)
        
        for agent_id, r in reward.items():
            agent_reward[agent_id] += r
        
        # learn according to learning frequency
        if (memory.counter % agent.learn_step == 0) and (len(memory) >= agent.batch_size):
            experiences = memory.sample(agent.batch_size)
            agent.learn(experiences)
            
        # update state
        state = next_state
        
        #break when all agents reach destination (doesnt work)
        '''
        _, done = env.state()
        
        if done == True:
            break
        '''
        
    # metric logging
    travel_time = env.state()
    episode_travel_times.append(travel_time) # export to csv
    travel_times_df.loc[len(travel_times_df)] = travel_time[0]
    
    # save the total episode reward
    score = sum(agent_reward.values())
    agent.scores.append(score)
    
    # update epsilon for exploration
    epsilon = max(eps_end, epsilon * eps_decay)
    
print(agent.scores) # export to csv
travel_times_df.to_csv('models/episode_travel_times.csv')