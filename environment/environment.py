import networkx as nx
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
import sys

from gymnasium import Discrete

sys.path.append("../network")

from test_net import test_net


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
        super(ParallelEnv).__init()
        
        # initialize on NetworkX graph
        self.road_network = graph
        
        # agent initialization
        self.agent_origins = {"agent_1": "A", "agent_2": "A"}
        self.agent_destinations = {"agent_1": "D", "agent_2": "D"}
        self.agent_positions = {agent: self.agent_origins[agent] for agent in self.agent_origins}
        
    def observe(self, agent):
        agent_positions = self.agent_positions[agent]
        agent_node_neighbors = list(self.road_network.neighbors(agent_positions))
    
    def step(self, action):
        pass
    
    def step(self, action):
        pass
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass