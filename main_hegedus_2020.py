from networkx import to_scipy_sparse_array
from networkx.generators.random_graphs import random_regular_graph
from gossipy import set_seed
from gossipy.core import UniformDelay, AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.node import GossipNode
from gossipy.model.handler import MFModelHandler
from gossipy.data import RecSysDataDispatcher, load_recsys_dataset
from gossipy.data.handler import  RecSysDataHandler
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

set_seed(98765)
ratings, nu, ni = load_recsys_dataset("ml-1m")
data_handler = RecSysDataHandler(ratings, nu, ni, .1, seed=42)
dispatcher = RecSysDataDispatcher(data_handler)
topology = StaticP2PNetwork(nu, to_scipy_sparse_array(random_regular_graph(20, nu, seed=42)))

dispatcher.assign()
nodes = GossipNode.generate(
    data_dispatcher=dispatcher,
    p2p_net=topology,
    model_proto=MFModelHandler(
        dim=5,
        n_items=ni,
        lam_reg=.1,
        learning_rate=0.001,
        create_model_mode=CreateModelMode.MERGE_UPDATE),
    round_len=100,
    sync=True)

simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    delay=UniformDelay(0, 10),
    sampling_eval=.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=100)

plot_evaluation([[ev for _, ev in report.get_evaluation(True)]], "User-wise test results")