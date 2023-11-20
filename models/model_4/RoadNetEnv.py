# RoadNetEnv

import gymnasium
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.spaces.utils import flatten_space
from gymnasium.utils import EzPickle
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from NguyenNetwork import nguyenNetwork, traffic, latency

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
        traffic = traffic(high=False),
        render_mode = None
        ):
        
        # no rendering at the moment
        self.render_mode = render_mode
        
        # initialize network from Nguyen Network
        self.road_network = net
        self.traffic = traffic
        
        # initialize agents
        self.agents = self.traffic["agents"] # list of agents in environment
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        self._agent_selector = agent_selector(self.agents)
        
        # check
        self.terminations_check = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations_check = dict(zip(self.agents, [False for _ in self.agents]))
        
        # agent origin, destination, and location information
        self.agent_origins = self.traffic["origins"].copy()
        self.agent_origin_backup = self.traffic["origins"].copy()
        self.agent_locations = self.traffic["origins"].copy()
        self.agent_destinations = self.traffic["destinations"]
        self.agent_path_histories = {agent: [location] for agent, location in zip(self.agents, self.agent_locations)}
        self.agent_wait_time = {agent: 0 for agent in self.agents}
        self.agent_travel_time = {agent: 0 for agent in self.agents}

        # agent unflattened observation space, this is flattened alphabetically btw.
        self.unflattened_observation_spaces = {
            agent: Dict({
                "observation": Box(low=-1, high=12, shape=(2,1), dtype=int),
                "latencies": Box(low=0, high=1e5, shape=(2,1), dtype=int),
                "location": Box(low = 0, high = 12, shape = (1,1), dtype=int),
            }) for agent in self.agents
        }
        
        # agent flattened observatino space
        self.observation_spaces = {
            i: flatten_space(self.unflattened_observation_spaces[i]) for i in self.unflattened_observation_spaces
        }
        
        # agent action space
        self.action_spaces = dict(
            # with the nguyen network agents have at most 2 choices
            zip(
                self.agents,
                [gymnasium.spaces.Discrete(2)]*len(self.agents)
            )
        )
        
        # agent terminal and truncated states
        self.terminate = False
        self.truncate = False
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):

        agent_idx = self.agent_name_mapping[agent]
        
        # get possible nodes the agent can travel to
        agent_position = self.agent_locations[agent_idx]
        agent_node_neighbors = list(self.road_network.neighbors(agent_position))
        
        # encode position
        position = []
        encoding = int(self.agent_locations[agent_idx])-1
        position.append(encoding)
        
        # encode node positions
        node_encoded = []
        for node in agent_node_neighbors:
            encoding = int(node)-1
            node_encoded.append(encoding)
        
        # currently using ffs attribute, need to revise to use updated latency
        # neighboring_nodes_ffs = []
        # for node in agent_node_neighbors:
        #     ffs_value = self.road_network.get_edge_data(agent_position, node)["ffs"]
        #     neighboring_nodes_ffs.append(ffs_value)

        # updated to use "latency" instead of "ffs"
        neighboring_nodes_latencies = []
        for node in agent_node_neighbors:
            latency_value = self.road_network.get_edge_data(agent_position, node)["latency"]
            neighboring_nodes_latencies.append(latency_value)

        # return observation – a list in structured as [node1, latency1, node 2, latency2]
        if len(agent_node_neighbors) == 1:
            node_encoded = node_encoded*2
            # neighboring_nodes_ffs = neighboring_nodes_ffs*2
            neighboring_nodes_latencies = neighboring_nodes_latencies*2
        if len(agent_node_neighbors) == 0:
            node_encoded = [-1,-1]
            # neighboring_nodes_ffs = [0,0]
            neighboring_nodes_latencies = [0,0]
            
        # observations  = np.array(neighboring_nodes_ffs+position+node_encoded)
        observations = np.array(neighboring_nodes_latencies + position + node_encoded)
        
        return observations



    def state(self) -> np.ndarray:
        """We need to return an np-array like object for logging"""
        return list(self.agent_travel_time.keys())



    def step(self, action):
        """This does not work"""
        # # check if agent is dead
        # if (
        #     self.terminations_check[self.agent_selection] or
        #     self.truncations_check[self.agent_selection]
        # ):
        #     # self.rewards[self.agent_selection] = 0
        #     self.agent_selection = self._agent_selector.next()
        #     #self._accumulate_rewards()
        #     return
        
        action = np.asarray(action)
        agent = self.agent_selection
        agent_idx = self.agent_name_mapping[agent]        

        self._cumulative_rewards[agent] = 0
        
        if self._agent_selector.is_last():
            # Track the count of agents on each link for all agents' positions
            agents_on_link = {edge: 0 for edge in self.road_network.edges()}
            # print(agents_on_link)
            # print(self.road_network.edges())

            # Update the count based on all agents' neighbors
            for current_agent in self.agents:                   
                agent_idx = self.agent_name_mapping[current_agent]                
                agent_position = self.agent_locations[agent_idx]                
                agent_node_neighbors = list(self.road_network.neighbors(agent_position))

                for node in agent_node_neighbors:                    
                    for edge in zip([agent_position], [node]):                        
                        agents_on_link[edge] += 1  # Increment count when an agent enters a link                                  

            for edge, agent_count in agents_on_link.items():                
                link_data = self.road_network.get_edge_data(*edge)  
                # print(link_data) # not updated                                       

                flow = agent_count  # 'flow' takes the total number of vehicles in the link                                         
                start_node = np.int64(edge[0])            
                end_node = np.int64(edge[1])            

                new_latency = latency(flow, start_node, end_node)                       
                link_data["latency"] = new_latency            
                # print(new_latency)                   

                self.road_network[edge[0]][edge[1]]["latency"] = new_latency
                # print(f"Updated latency for edge {edge}: {self.road_network[edge[0]][edge[1]]['latency']}") 

            self._clear_rewards()
        else:
            self._clear_rewards()
            pass

        # agent travel decrement
        if self.agent_wait_time[agent] != 0:
            # if agent has waiting time (i.e. "traveling" along edge, decrement wait time by one time step)
            self.agent_wait_time[agent] -= 1
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return
        
        # agent reaches terminal state
        if self.agent_locations[agent_idx] == self.agent_destinations[agent_idx]:
            self.terminations_check[agent] = True
            
            # return reward for arriving at destionation
            completion_reward = 0 # this value may be adjusted in the future
            self.rewards[agent] = completion_reward
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return
        
        # agent reaches truncation state
        if self.agent_locations[agent_idx] != self.agent_destinations[agent_idx] and \
        (self.agent_locations[agent_idx] == "2" or self.agent_locations[agent_idx] == "3"):
            self.truncations_check[agent] = True
                    
            # return penalty for arriving at wrong destination
            completion_penalty = -100 # this value may be adjusted in the future
            self.rewards[agent] = completion_penalty
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return
        
        # agent chooses action
        choices =  list(
            self.road_network.neighbors(
                self.agent_locations[agent_idx]
            )
        )
        
        # if only one route
        if len(choices) == 1:
            choices = [choices[0], choices[0]]
        
        # select action
        chosen_route = choices[action]

        # reward based on chosen route latency, again using ffs instead of calculated latency, need a _calculate_reward(agent) method for this
        reward = self.road_network.get_edge_data(
            self.agent_locations[agent_idx],
            # chosen_route)["ffs"]
            chosen_route)["latency"]
        
        # add negative latency to reward – DQN to maximize negative reward
        self.rewards[agent] = -1*reward
        
        # update latency
        self.agent_wait_time[agent] += reward
        self.agent_travel_time[agent] += reward        
        
        # update agent position
        self.agent_locations[agent_idx] = chosen_route
        
        # update path history
        self.agent_path_histories[agent].append(chosen_route)
        
        
        # set the next agent to act
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()


    def reset(self, seed=None, options=None):
        # reset to initial states
        self.agent_origins = self.agent_origin_backup.copy()
        self.agent_locations = self.agent_origin_backup.copy()
        self.agent_path_histories = {agent: [location] for agent, location in zip(self.agents, self.agent_origins)}
        self.agent_wait_time = {agent: 0 for agent in self.agents}
        self.agent_travel_time = {agent: 0 for agent in self.agents} # added

        self.agents = self.possible_agents[:]        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        
        # Reset termination and truncation checks
        # check
        self.terminations_check = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations_check = dict(zip(self.agents, [False for _ in self.agents]))
        
        # Reset road network-related variables
        # print("Resetting latency values")
        for edge in self.road_network.edges():
            start_node, end_node = edge
            self.road_network[start_node][end_node]["latency"] = 0            

        # Clear any existing rewards
        #self._clear_rewards()
        # print(self.rewards)
        # print(self.state())

        # Return the initial state
        #return self.state()
        

