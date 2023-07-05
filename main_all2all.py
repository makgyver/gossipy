import networkx as nx
from networkx.generators import random_regular_graph
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from gossipy import GlobalSettings, set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformMixing
from gossipy.node import All2AllGossipNode
from gossipy.model.handler import WeightedTMH
from gossipy.model.nn import LogisticRegression
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import All2AllGossipSimulator, SimulationReport
from gossipy.utils import plot_evaluation

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2023, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

set_seed(98765)
#GlobalSettings().set_device("mps")
X, y = load_classification_dataset("spambase", as_tensor=True)
data_handler = ClassificationDataHandler(X, y, test_size=.1)
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False, auto_assign=True)
topology = StaticP2PNetwork(dispatcher.size(), topology=nx.to_numpy_array(random_regular_graph(20, 100, seed=42)))
net = LogisticRegression(data_handler.Xtr.shape[1], 2)

nodes = All2AllGossipNode.generate(
    data_dispatcher=dispatcher,
    p2p_net=topology,
    model_proto=WeightedTMH(
        net=net,
        optimizer=torch.optim.SGD,
        optimizer_params={
            "lr": .1,
            "weight_decay": .01
        },
        criterion=CrossEntropyLoss(),
        create_model_mode=CreateModelMode.MERGE_UPDATE),
    round_len=100,
    sync=False
)

simulator = All2AllGossipSimulator(
    nodes=nodes,
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(UniformMixing(topology), n_rounds=100)

plot_evaluation([[ev for _, ev in report.get_evaluation(local=False)]], "Overall test results")
