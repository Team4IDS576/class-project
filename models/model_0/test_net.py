import networkx as nx
import matplotlib.pyplot as plt

def test_net():
    
    # instantiate null directed graph
    network = nx.DiGraph()
    
    # initialize intersections
    intersections = {
        "A": {"pos": (0, 0)},
        "B": {"pos": (2, 0)},
        "C": {"pos": (0, 2)},
        "D": {"pos": (2, 2)}
    }

    # add intersections
    for node, attrs in intersections.items():
        network.add_node(node, **attrs)
    
    # intialize roads
    roads = [
        ("A", "B", {"length": 2, "congestion": 2}),
        ("B", "D", {"length": 2, "congestion": 2}),
        ("A", "C", {"length": 2, "congestion": 2}),
        ("C", "D", {"length": 2, "congestion": 2})
    ]

    # add roads
    network.add_edges_from(roads)
    
    return network

if __name__ == "__main__":
    
    # load test net
    network = test_net()
    
    # get position of nodes
    pos = nx.get_node_attributes(network, "pos")
    
    # graph network
    plt.figure(figsize=(6,6))
    nx.draw(network, pos, with_labels=True)
    plt.axis('off')
    plt.show()
