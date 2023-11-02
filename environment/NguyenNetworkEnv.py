# Traffic Assignment Network

"""
The structure of this class is loosely based off of the PettinZoo Pistonball environment https://pettingzoo.farama.org/_modules/pettingzoo/butterfly/pistonball/pistonball/#raw_env

While this is a parallel environment the raw env is actually an AEC env with a parallel wrapper
"""

# gymnasium imports
import gymnasium
import gymnasium.spaces
import numpy as np
from gymnasium.utils import EzPickle

# petting zoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

# network imports
import sys
import networkx as nx
sys.path.append("../network/")
from NguyenNetwork import nguyenNetwork, latency


# traffic imports
sys.path.append("../traffic")

# allows to import the parallel environment using "from NguyenNetworkEnv import parallel_env"
__all__ = ["parallel_env"]

def env(**kwargs):
    env = raw_env(**kwargs)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    metadata = {
        "name": "NguyenNet"
    }
    
    # initialize environment with the Nguyen Network and Traffic Demand from agents
    def __init__(self, net, traffic):
        '''
        net:
        '''
        # initialize network
        self.road_network = net
        
        # initialize agents, orgins, current positions, and destinations
        self.agents = traffic["agents"] # list of agents in environment
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents))))) # map list of agents to int id starting at 0
        self._agent_selector = agent_selector(self.agents)
        
        # agent travel information
        self.agent_origins = traffic["origins"]
        self.agent_locations = traffic["origins"]
        self.agent_destinations = traffic["destinations"]
        
        # agent wait times initialized at zero
        self.agent_wait_time = {agent: 0 for agent in self.agents}
        
        # agent observation space
        self.observation_spaces = dict(
            # dict of agents and there observation spaces - at most 4 corresponding to two possible choices and there latencies
            zip(
                self.agents,
                gymnasium.spaces.Discrete(4)
            )
        )
        
        # agent action space
        self.action_spaces = {
            # with the nguyen network agents have at 2 nodes to travel to
            zip(
                self.agents,
                gymnasium .spaces.Discrete(2)
            )
        }
        
        # current agent
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # agent terminal and truncated state
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        
        # agent rewards
        self.rewards = {agent: 0 for agent in self.agents}
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        
        # get possible nodes the agent can travel to
        agent_position = self.agent_locations[agent]
        agent_node_neighbors = list(self.road_network.neighbors(agent_position))
        
        # currently using ffs attribute, need to revise to use updated latency
        neighboring_nodes_ffs = []
        for node in agent_node_neighbors:
            ffs_value = self.road_network.nodes[node]["ffs"]
            neighboring_nodes_ffs.append(ffs_value)
        
        # return observation – a list in structured as [node1, latency1, node 2, latency2]
        if len(agent_node_neighbors) == 1:
            return [val for pair in zip(agent_node_neighbors, neighboring_nodes_ffs) for val in pair] * 2
        else:
            return [val for pair in zip(agent_node_neighbors, neighboring_nodes_ffs) for val in pair]

    def state(self) -> np.ndarray:
        pass
    
    def step(self, action):
        # agent is dead pass
        if (self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
            ):
            return
        
        # need to add logic to update network
        
        # select agent
        agent = self.agent_selection
        
        if self.agent_wait_time[self.agent_selection] != 0:
            # if agent has waiting time (i.e. "traveling" along edge, decrement wait time by one time step)
            self.agent_wait_time[self.agent_selection] -= 1
            return
        else:
            # select node to move to from list of available nodes
            chosen_route = list(self.road_network.neighbors(self.agent_locations[agent]))[action]
            
            # reward based on chosen route latency, again using ffs instead of calculated latency, need a _calculate_reward(agent) method for this
            reward = self.road_network.get_edge_data(self.agent_locations[agent], chosen_route)["ffs"]
            self.rewards[agent] += reward
            
            # update agent position
            self.agent_locations[agent] = chosen_route
            
            # kill agent if reached destination
            if chosen_route == self.agent_destinations[agent]:
                self.terminations[agent] == True
            
            # set the next agent to act
            self.agent_selection = self._agent_selector.next()
            
            return self.observe(self.agent_selection), reward, self.terminations[agent], {}