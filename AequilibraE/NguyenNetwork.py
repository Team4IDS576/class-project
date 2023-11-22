import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def nguyenNetwork(high=True):
    
    # Read the CSV file into a DataFrame
    data_types = {"start node": str, "end node": str} # node names should be str
    
    # for mac OS
    if high == True:
        links = pd.read_csv("./AequilibraE/newNguyenLinksHighDemand.csv", dtype=data_types)
    else:
        links = pd.read_csv("./AequilibraE/newNguyenLinksHighDemand.csv", dtype=data_types)
    
    links['flow'] = 0
    
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
               "beta": row["beta"],
               "latency": row["latency"],
               "flow": row["flow"]}) for _, row in links.iterrows()]

    # Print the list of roads
    # print(roads)


    # add roads
    network.add_edges_from(roads)
    
    return network

def latency(flow, start_node, end_node):
    network = nguyenNetwork()
    edge_data = network.get_edge_data(str(start_node), str(end_node))
    
    # 'flow' takes the total No. of vehicle in the link
    # 'start_node' and 'end_node' should be integer values
    # This function assumes that the travel time of the links are BPR functions.
    # for more information about BPR functions read p#358 of Sheffi's book.
    c = edge_data['capacity']
    t_0 = edge_data['ffs']
    a = edge_data['alpha']
    b = edge_data['beta']
    t_link = t_0 * (1 + (a * ((flow/c) ** b)))
    return round(t_link) # this function only returns integer values


def traffic(high=True):
    if high == True:
        demand = pd.read_csv("./AequilibraE/newNguyenDemandHighDemand.csv")
    else:
        demand = pd.read_csv("./AequilibraE/newNguyenDemandLowDemand.csv")
    agents = []
    origins = []
    destinations = []
    agent_no = 0

    for index, row in demand.iterrows():
        origin = str(row['Origin'])
        destination = str(row['Destination'])
        count = int(row['OD demand'])
        for i in range(count):
            agent_no += 1
            agents.append(f'agent_{agent_no}')
            origins.append(origin)
            destinations.append(destination)

    traffic = {
        "agents": agents,
        "origins": origins,
        "destinations": destinations
    }

    return traffic



if __name__ == "__main__":
    
    import os
    
    print(os.getcwd())
    
    # Read the CSV file into a DataFrame
    data_types = {"start node": str, "end node": str} # node names should be str

    # for mac OS
    links = pd.read_csv("./AequilibraE/newNguyenLinksHighDemand.csv", dtype=data_types)
    demand = pd.read_csv("./AequilibraE/newNguyenDemandHighDemand.csv", dtype=data_types)

    # for windows
    #links = pd.read_csv("NguyenLinks.csv", dtype=data_types)
    #demand = pd.read_csv("NguyenDemand.csv")

    # load test net
    network = nguyenNetwork()
    
    # get position of nodes
    pos = nx.get_node_attributes(network, "pos")
    
    # graph network
    plt.figure(figsize=(6,6))
    nx.draw(network, pos, with_labels=True)

    # Create a mapping from (start_node, end_node) to Link No
    link_mapping = {(str(row['start node']), str(row['end node'])): str(row['No'])
                for index, row in links.iterrows()}

    # Annotate edges with the correct link numbers from the CSV
    for edge in network.edges():
        # Find the midpoint of the edge for placing the text label
        edge_midpoint = ((pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                         (pos[edge[0]][1] + pos[edge[1]][1]) / 2)
        # Get the correct link number from the mapping
        link_no = link_mapping.get(edge) or link_mapping.get((edge[1], edge[0]))  # Check both directions
        # Place the link number as a text label on the midpoint of the edge
        plt.text(edge_midpoint[0], edge_midpoint[1], link_no, fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))


    plt.axis('off')
    plt.show()

    # Example usage
    result = traffic()
    print(type(result))