from scipy.spatial.distance import euclidean, cosine
import numpy as np
import pandas as pd

class edge:
    '''
    This class represents the edges (road segments) of the network
    '''
    def __init__(self, id: int, v1: tuple = (0,0), v2: tuple = (1,1), speed_limit: int = 0):
        # initialized attributes
        self.id = id # int id of edge
        self.v1 = np.array(v1) # first (start) vertex (x, y) of segment in feet
        self.v2 = np.array(v2) # second (end) vertext (x, y) of segment in feet
        self.speed_limit = speed_limit # speed limit of segment feet/sec (need a conversion)
        
        # calculated attributes
        self.length = euclidean(self.v1, self.v2) # length of segment in feet
        self.direction = self.v2 - self.v1 # get direction of the road
        self.angle = np.arctan2(self.direction[0], self.direction[1]) # angle of segment in radians
        self.angle_degrees = np.degrees(self.angle) # angle in degrees
        self.unit_vector = self.direction / self.length # unit vector in direction of
        
        # road properties
        '''
        These attributes are related to the modeling of traffic flow
        '''
        self.d1 = self.unit_vector # unit step in the direcition of v1 to v2
        self.d2 = -1 * self.unit_vector # unit step in the direction of v2

    def info(self):
        '''
        This method returns a list of attributes of the edge
        '''
        return [self.id, self.length, self.angle_degrees, self.speed_limit]

class node:
    '''
    This class represents the node (intersection) components of the network
    '''
    def __init__(self, id: int, location: tuple):
        # initialized attributes
        self.id = id # int id of intersection
        self.location = np.array(location) # location of intersection (x, y) in feet
        
        # intersection properties
        '''
        These attributes store the id's of edges connected to the node and the directionality of the edge
        (whether v1 or v2 is connected to the node).
        '''
        self.edge_ids = [] # integer id of edge connected to node. Assuming four edges per node
        self.edge_directions = [] # bool value such that True = connected to v1 and False = connected to v2 of edge.

    def add_edge(self, edge: edge):
        '''
        This method adds an edge and keeps track of the end attached to the node.
        '''
        # check to make sure edge can connect to node
        if np.array_equal(self.location, edge.v1) or np.array_equal(self.location, edge.v2):
            # append edge id
            self.edge_ids.append(edge.id)
            
            # append direction of edge based on start vertex v1
            self.edge_directions.append(
                np.array_equal(self.location, edge.v1)
            )
        else:
            pass

    def summary(self):
        '''
        This is a debugging function that prints a summary of the node
        '''
        print(f"Node id: {self.id}\nConnected Edges:")
        for edge_id, direction in zip(self.edge_ids, self.edge_directions):
            vertex = "v1" if direction else "v2"
            print(f"edge id: {edge_id}\tVertex: {vertex}")

class roadnet:
    ''''
    This class builds a road network.
    
    begin by instantiating a roadnet object, add nodes and edges or list of nodes and edges
    
    use the build() method to connect edges to nodes
    
    
    '''
    def __init__(self):
        self.edges = [] # list of edges
        self.nodes = [] # list of nodes

    def add_edge(self, edge: edge):
        self.edges.append(edge) # append to edges instance variable

    def add_edges(self, edges: [edge]):
        for edge in edges:
            self.add_edge(edge) # for list of edges call add_edge() method

    def get_edge(self, id: int):
        for edge in self.edges:
            if id == edge.id:
                return edge
        return None # return None if not in edges
            
    def add_node(self, node):
        self.nodes.append(node)

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def get_node(self, id):
        for node in self.nodes:
            if id == node.id:
                return node
        return None # return None if not in nodes