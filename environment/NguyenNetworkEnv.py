# Traffic Assignment Network

"""
The structure of this class is loosely based off of the PettinZoo Pistonball environment https://pettingzoo.farama.org/_modules/pettingzoo/butterfly/pistonball/pistonball/#raw_env

While this is a parallel environment the raw env is actually an AEC env with a parallel wrapper
"""

# gymnasium imports
import gymnasium
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
        
        # initialize agents, orgins, current positions, and destinations
        self.agents = traffic["agents"] # list of agents in environment
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents))))) # map list of agents to int id starting at 0
        self._agent_selector = agent_selector(self.agents)
        
        # agent location infomration
        self.agent_origins = traffic["origins"]
        self.agent_locations = traffic["origins"]
        self.agent_destinations = traffic["destinations"]
        