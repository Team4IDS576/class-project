from scipy.spatial.distance import euclidean, cosine
import numpy as np

class edge:
    def __init__(self, id: int, v1: tuple, v2: tuple, speed_limit: int):
        
        # initialized attributes
        self.id = id # int id of edge
        self.v1 = np.array(v1) # first vertex (x, y) of segment in feet
        self.v2 = np.array(v2) # second vertext (x, y) of segment in feet
        self.speed_limit = speed_limit # speed limit of segment
        
        # calculated attributes
        self.length = self._calc_length() # length of segment in feet
        self.angle = self._calc_angle() # angle of segment in degrees
    
    def _calc_length(self):
        return euclidean(self.v1, self.v2)
    
    def _calc_angle (self):
        direction_vector = self.v2 - self.v1 # calculate the direction of the segment
        
        # return arctan2 of direction vector
        return np.degrees(np.arctan2(direction_vector[0], direction_vector[1]))
        