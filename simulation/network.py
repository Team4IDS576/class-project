from scipy.spatial.distance import euclidean, cosine
import numpy as np

class edge:
    def __init__(self, id: int, v1: tuple, v2: tuple, speed_limit: int):
        """
        This functinon initializes a segment in the roadway network
        """
        
        # initialized attributes
        self.id = id # int id of edge
        self.v1 = np.array(v1) # first vertex (x, y) of segment in feet
        self.v2 = np.array(v2) # second vertext (x, y) of segment in feet
        self.speed_limit = speed_limit # speed limit of segment
        
        # calculated attributes
        self.length = euclidean(self.v1, self.v2) # length of segment in feet
        self.direction = self.v2 - self.v1 # get direction of the road
        self.angle = np.arctan2(self.direction[0], self.direction[1]) # angle of segment in radians
        self.angle_degrees = np.degrees(self.angle)
        self.unit_vector = self.direction / self.length
        
        # road properties
        '''
        These attributes are related to the modeling of traffic flow
        '''
        self.d1 = self.unit_vector # unit step in the direcition of v1
        self.d2 = -1 * self.unit_vector # unit step in the direction of v2
