import networkx as nx
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
import sys

from gymnasium.spaces import Discrete

sys.path.append("../network")

from old.test_net import test_net


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = RoadNetworkEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class RoadNetworkEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, graph):
        #super().__init()
        
        # initialize on NetworkX graph
        self.road_network = graph
        
        # agent initialization
        self.agent_origins = {"agent_1": "A", "agent_2": "A"}
        self.agent_destinations = {"agent_1": "D", "agent_2": "D"}
        self.agent_positions = self.agent_origins
        
    def observe(self, agent):
        agent_position = self.agent_positions[agent]
        agent_node_neighbors = list(self.road_network.neighbors(agent_position))
        congestion_info = {neighbor: self.road_network[agent_position][neighbor]['congestion'] for neighbor in agent_node_neighbors}
        agent_observations = {
            "position": agent_position,
            "neighbors": agent_node_neighbors,
            "congestion": congestion_info
        }
        
        return agent_observations
    
    def step(self, action, agent):
        current_position = self.agent_positions[agent]
        
        if action in self.road_network.neighbors(current_position):
            self.agent_positions[agent] = action
            
            reward = 0  # define the reward logic for the agent
            done = self.agent_positions[agent] == self.agent_destinations[agent]
            info = {}  # additional information
            
            # Return the results for the agent
            return self.observe(agent), reward, done, info
        else:
            return self.observe(agent), -1, False, {}

    def reset(self):
        # Reset agent positions to their origins
        self.agent_positions = self.agent_origins.copy()
        
        # Return initial observations for each agent
        observations = {agent: self.observe(agent) for agent in self.agent_positions}
        return observations
        
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass