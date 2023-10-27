import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class RoadNetworkEnv(gym.Env):
    def __init__(self, graph):
        super(RoadNetworkEnv, self).__init__()

        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.action_space = spaces.Discrete(len(self.nodes))
        self.observation_space = spaces.Discrete(len(self.nodes))
        self.current_node = np.random.choice(self.nodes)

    def reset(self):
        # Reset to a random node
        self.current_node = np.random.choice(self.nodes)
        return self.current_node

    def step(self, action):
        reward = self._calculate_reward(action)
        done = False  # Termination condition

        # Update the current node based on the action (movement to a connected node)
        if action < len(self.nodes):
            possible_moves = list(self.graph.neighbors(self.current_node))
            if self.nodes[action] in possible_moves:
                self.current_node = self.nodes[action]

        return self.current_node, reward, done, {}

    def _calculate_reward(self, action):
        # Custom reward function based on specific criteria
        # For instance, you might want to encourage reaching a certain destination node
        # or penalize for taking certain paths.

        # Example: reward based on proximity to a target node
        target_node = "Destination_Node"
        if self.nodes[action] == target_node:
            return 10  # High reward for reaching the destination
        else:
            return -1  # Small penalty for moving to a non-target node

    def render(self, mode='human'):
        # You can implement a rendering method to visualize the current state if needed.
        pass

# Example usage:

# Create a graph (you can define your own road network using NetworkX or any other graph library)
G = nx.Graph()
G.add_nodes_from(["A", "B", "C", "D"])
G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])

# Create an instance of the environment
env = RoadNetworkEnv(G)

# Example of using the environment
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Taking a random action
    obs, reward, done, _ = env.step(action)
    print(f"Current Node: {obs}, Reward: {reward}")
