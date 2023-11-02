import networkx as nx
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete

import sys
sys.path.append("../network")

from NguyenNetwork import nguyenNetwork, latency

class ParallelRoadNetEnv(ParallelEnv):
    metadata = {
        "name": "NguyenDupuis Net",
        "is_parallelizable": True
    }
    
    def __init__(self, net):