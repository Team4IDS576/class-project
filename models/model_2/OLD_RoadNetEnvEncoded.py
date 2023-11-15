# RoadNetEnv

# gymnasium imports
import gymnasium
import gymnasium.spaces
from gymnasium.utils import EzPickle
import numpy as np

# petting zoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

# network imports
from NguyenNetwork import nguyenNetwork, traffic

# encoding
from sklearn.preprocessing import LabelBinarizer

# allows to import the parallel environment using "from NguyenNetworkEnv import parallel_env"
__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]

# environment wrapper
def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# AEC to parallel wrapper
parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    metadata = {
        "name": "NguyenNet",
        "is_parallelizable": True
    }
    
    """
    This is the traffic assignment environment. More documentation to follow.
    """
    
    # initialize environment with the Nguyen Network and Traffic Demand from agents
    def __init__(
        self,
        net = nguyenNetwork(),
        traffic = traffic(),
        render_mode = None
        ):
        
        '''
        net: NetworkX directed graph network.
        traffic: Traffic volumes, origins, destinations, and other parameters passed as a dict.
        '''
        
        # no rendering at the moment
        self.render_mode = render_mode
        
        # initialize network from Nguyen Network
        self.road_network = net
        self.traffic = traffic
        
        # initialize agents
        self.agents = self.traffic["agents"] # list of agents in environment
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(
                self.agents,
                list(range(len(self.agents)))
                )
            )
        #self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        
        """
        We currently have two ways of indexing agents, either there
        string "Agent_ID" name or there integer id from agent_name_mapping.
        We should probably refactor to use one or the other in the future.
        """
        
        # agent origin, destination, and location information
        self.agent_origins = self.traffic["origins"]
        self.agent_locations = self.traffic["origins"]
        self.agent_destinations = self.traffic["destinations"]
        
        # store agent path history as a lists
        self.agent_path_histories = {agent: [location] for agent, location in zip(self.agents, self.agent_origins)}
        
        # agent wait times initialized at zero
        self.agent_wait_time = {agent: 0 for agent in self.agents}
        
        # agent observation space
        """
        dict of agents and there observation spaces - at most 4 corresponding to two possible choices and there latencies.
        """
        self.observation_spaces = dict(
            zip(
                self.agents,
                [gymnasium.spaces.Discrete(28)]*len(self.agents)
            )
        )
        
        # agent action space
        self.action_spaces = dict(
            # with the nguyen network agents have at most 2 nodes to travel to
            zip(
                self.agents,
                [gymnasium.spaces.Discrete(2)]*len(self.agents)
            )
        )
        
        # agent terminal and truncated states
        self.terminate = False
        self.truncate = False
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        
        # agent rewards
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        
        # encoding 
        nodes = list(self.road_network.nodes())
        lb = LabelBinarizer()
        one_hot_encoded = lb.fit_transform(nodes)
        
        self.location_mapping = dict(
            zip(
                nodes, one_hot_encoded
            )
        )
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        
        agent_idx = self.agent_name_mapping[agent]
        
        # get possible nodes the agent can travel to
        agent_position = self.agent_locations[agent_idx]
        agent_node_neighbors = list(self.road_network.neighbors(agent_position))
        
        # encode node positions
        node_encoded = []
        for node in agent_node_neighbors:
            encoding = self.location_mapping[node]
            node_encoded.append(encoding)
        
        # currently using ffs attribute, need to revise to use updated latency
        neighboring_nodes_ffs = []
        for node in agent_node_neighbors:
            ffs_value = self.road_network.get_edge_data(agent_position, node)["ffs"]
            neighboring_nodes_ffs.append(ffs_value)
        
        # return observation – a list in structured as [node1, latency1, node 2, latency2]
        if len(agent_node_neighbors) == 1:
            return [list(node_encoded[0])+[neighboring_nodes_ffs[0]]] * 2
        elif len(agent_node_neighbors) == 2:
            return list(node_encoded[0])+[neighboring_nodes_ffs[0]]+list(node_encoded[1])+[neighboring_nodes_ffs[1]]
        else:
            return [0]*28

    def state(self) -> np.ndarray:
        "We need to return an np-array like object for logging"
        pass
    
    def step(self, action):
        
        
        """
        This logic should be documented:
        1) select agent
        2) check if agent is terminated or truncated - if so pass
        3) If agent has selected a route and is "traveling", decrement by one time step
        3a) If agent wait time is zero agent has reached a node and will select next edge to travel on
        3b) If only one node to travel to, action is 
        """
        
        # select next agent
        agent = self.agent_selection
        agent_idx = self.agent_name_mapping[agent]
        
        # print(self.terminations)
        # print(agent)
        # print(self.truncations[agent])
        # print(self.terminations[agent])
        # print(self.agent_path_histories[agent])
        # for edge in self.road_network.edges(data=True):
        #     source, target, attributes = edge
        #     print(f"Edge: {source} -> {target}, Attributes: {attributes}")
        
        # need to add logic to update network – I dont't know if this should be done on before first agent or last agent
        
        """
        1) look self.agent_path_histories
        2) get the last two items in each list of path histories
        3) get the number of agents that are on each link
        4) update the latency based on the number of agents in the link
        5) update the network using self.road_network.get_edge_data("node1","node2")["latency"] = new latency using latency function in NguyenNetwork.py
        
        """
        
        # agent travel decrement
        if self.agent_wait_time[agent] != 0:
            # if agent has waiting time (i.e. "traveling" along edge, decrement wait time by one time step)
            self.agent_wait_time[agent] -= 1
            self.agent_selection = self._agent_selector.next()
            return
        
        else:

            # if agent is dead on previous step pass
            if (self.terminations[agent]
            or self.truncations[agent]):
                self.agent_selection = self._agent_selector.next()
                return
            
            # kill agent if arrived at destination            
            if self.agent_locations[agent_idx] ==\
                self.agent_destinations[agent_idx]:
                    self.terminations[agent] = True
                    
                    # return reward for arriving at destionation
                    reward = 0 # this value may be adjusted in the future
                    self.agent_selection = self._agent_selector.next()
                    return self.observe(self.agent_selection), reward, self.terminations[agent], {}
            
            # truncate agent if arrived at wrong node
            if self.agent_locations[agent_idx] !=\
                self.agent_destinations[agent_idx] and (
                self.agent_locations[agent_idx] == "2" or
                self.agent_locations[agent_idx] == "3"):
                    self.truncations[agent] = True
                    
                    # return penalty for arriving at wrong destination
                    reward = 0 # this value may be adjusted in the future
                    self.agent_selection = self._agent_selector.next()
                    return self.observe(self.agent_selection), reward, self.terminations[agent], {}
            
            # select node to move to from list of available nodes
            choices = list(
                self.road_network.neighbors(
                    self.agent_locations[agent_idx]
                    )
                )
            
            # if only one route 
            if len(choices) == 1:
                choices = [choices[0], choices[0]]

            chosen_route = choices[action] 

            # reward based on chosen route latency, again using ffs instead of calculated latency, need a _calculate_reward(agent) method for this
            reward = self.road_network.get_edge_data(
                self.agent_locations[agent_idx],
                chosen_route)["ffs"]
            
            # add negative latency to reward – DQN to maximize negative reward
            self.rewards[agent] -= reward
            
            # update latency
            self.agent_wait_time[agent] += reward
            
            # update agent position
            self.agent_locations[agent_idx] = chosen_route
            
            # update path history
            self.agent_path_histories[agent].append(chosen_route)
            
            # set the next agent to act
            self.agent_selection = self._agent_selector.next()
            
            return self.observe(self.agent_selection), reward, self.terminations[agent], self.truncations[agent], {}

    def reset(self, *, seed=None, options=None):
        
        # reset to initial states
        self.agents = self.possible_agents[:]
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.terminate = False
        self.truncate = False
        
        self.agent_locations = self.agent_origins.copy()
        self.agent_path_histories = {agent: [location] for agent, location in zip(self.agents, self.agent_origins)}
        self.agent_wait_time = {agent: 0 for agent in self.agents}

        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
                
        # we will also need to reset the network - to be added

        # return initial observations for each agent
        #initial_observations = {agent: self.observe(agent) for agent in self.agents}
        #return initial_observations, {}