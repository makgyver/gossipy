import networkx as nx
from networkx.generators.trees import random_tree
from networkx.generators import barabasi_albert_graph
from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import AdaLine
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
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

set_seed(42)
X, y = load_classification_dataset("spambase", as_tensor=True)
y = 2*y - 1 #convert 0/1 labels to -1/1
data_handler = ClassificationDataHandler(X, y, test_size=.1)

data_dispatcher = DataDispatcher(data_handler, eval_on_user=False, auto_assign=True)
topology = StaticP2PNetwork(data_dispatcher.size(), topology=nx.to_numpy_array(barabasi_albert_graph(data_handler.size(), 10)))

nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=PegasosHandler(
        net=AdaLine(data_handler.size(1)),
        learning_rate=.01,
        create_model_mode=CreateModelMode.MERGE_UPDATE),
    round_len=100,
    sync=False
)

simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=100)

plot_evaluation([[ev for _, ev in report.get_evaluation(local=False)]], "Overall test results")
