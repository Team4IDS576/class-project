import os
import pandas as pd
import numpy as np
from tempfile import gettempdir

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
#The below code is changing the current working directory to the path mentioned.
#You can always change this path based on your local setup:
os.chdir("C:/github/Class-Project/class-project/AequilibraE")

folder = gettempdir()

dem = pd.read_csv("newNguyenDemandHighDemand.csv")
zones = int(max(dem.Origin.max(), dem.Destination.max()))
index = np.arange(zones) + 1

mtx = np.zeros(shape=(zones, zones))
for element in dem.to_records(index=False):
    mtx[element[0]-1] [element[1]-1] = element[2]


#aemfile = os.path.join(folder, "demand.aem")
aemfile = "C:\github\Class-Project\class-project\AequilibraE\demand.aem"
aem = AequilibraeMatrix()

aem.create_empty(file_name=aemfile, zones=zones, matrix_names=['matrix'], memory_only=False)
aem.matrix['matrix'][:,:] = mtx[:,:]
aem.index[:] = index[:]

net = pd.read_csv("newNguyenLinksHighDemand.csv", sep=",", lineterminator="\n")

net.columns = ["newline", "a_node", "b_node", "free flow time", "capacity", "alpha", "beta", "latency"]

network = net[["a_node", "b_node", "free flow time", "capacity", "alpha", "beta"]]
network = network.assign(direction=1)
network["link_id"] = network.index + 1
network = network.astype({"a_node":"int64", "b_node": "int64"})

g = Graph()
g.cost = network['free flow time'].values
g.capacity = network['capacity'].values
g.free_low_time = network['free flow time'].values

g.network = network
g.network_ok = True
g.status = 'OK'
g.prepare_graph(index)
g.set_graph("free flow time")
g.cost = np.array(g.cost, copy=True)
g.set_skimming(["free flow time"])
g.set_blocked_centroid_flows(False)
g.network["id"] = g.network.link_id

aem = AequilibraeMatrix()
aem.load(aemfile)
aem.computational_view(["matrix"])

assigclass = TrafficClass("car", g, aem)

assig = TrafficAssignment()

assig.set_classes([assigclass])
assig.set_vdf("BPR")
assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})
assig.set_capacity_field("capacity")
assig.set_time_field("free flow time")
assig.set_algorithm("fw")
assig.max_iter = 100
assig.rgap_target = 1e-6
assig.execute()


results=assig.results()
results.to_csv("newResults_HighDemand.csv")
print(results)

report = assig.report()
report.to_csv("newReport_HighDemand.csv")
print(report)