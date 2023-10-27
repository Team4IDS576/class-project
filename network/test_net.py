import networkx as nx
import matplotlib.pyplot as plt

def test_net():
    
    # instantiate null graph
    network = nx.DiGraph()
    
    # initialize intersections
    intersections = {
        "A": (0, 0),
        "B": (2, 0),
        "C": (0, 2),
        "D": (2, 2)
    }
    
    # add intersections
    network.add_nodes_from(intersections)
    
    # intialize roads
    roads = [("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")]
    
    # add roads
    network.add_edges_from(roads)
    
    return network

if __name__ == "__main__":
    pass