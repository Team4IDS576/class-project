import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# Read the CSV file into a DataFrame
data_types = {"start node": str, "end node": str} # node names should be str
links = pd.read_csv("NguyenLinks.csv", dtype=data_types)


def nguyenNetwork(links=links):
    
    # instantiate null directed graph
    network = nx.DiGraph()
    
    # initialize intersections
    intersections = {
        "1": {"pos": (1, 3)},
        "2": {"pos": (4, 1)},
        "3": {"pos": (3, 0)},
        "4": {"pos": (0, 2)},
        "5": {"pos": (1, 2)},
        "6": {"pos": (2, 2)},
        "7": {"pos": (3, 2)},
        "8": {"pos": (4, 2)},
        "9": {"pos": (1, 1)},
        "10": {"pos": (2, 1)},
        "11": {"pos": (3, 1)},
        "12": {"pos": (2, 3)},
        "13": {"pos": (2, 0)}
    }

    # add intersections
    for node, attrs in intersections.items():
        network.add_node(node, **attrs)
    
    # intialize roads

    # Create a list of tuples from the DataFrame
    roads = [(row["start node"], 
              row["end node"], 
              {"ffs": row["free flow speed"], 
               "capacity": row["capacity"],
               "alpha": row["alpha"],
               "beta": row["beta"]}) for _, row in links.iterrows()]

    # Print the list of roads
    print(roads)


    # add roads
    network.add_edges_from(roads)
    
    return network



def latency(flow, link):
    # flow takes the total No. of vehicle in the link
    # link takes the # of the link ("No" dolumn in the CSV file of the network)
    c = links[links['No'] == link]['capacity']
    t_0 = links[links['No'] == link]['free flow speed']
    a = links[links['No'] == link]['alpha']
    b = links[links['No'] == link]['beta']
    t_link = t_0 * (1 + (a * ((flow/c) ** b)))
    return t_link


if __name__ == "__main__":
    
    # load test net
    network = nguyenNetwork()
    
    # get position of nodes
    pos = nx.get_node_attributes(network, "pos")
    
    # graph network
    plt.figure(figsize=(6,6))
    nx.draw(network, pos, with_labels=True)
    plt.axis('off')
    plt.show()
