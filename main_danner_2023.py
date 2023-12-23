import torch
from networkx import to_numpy_array
from networkx.generators.random_graphs import random_regular_graph
from gossipy import set_seed
from gossipy.core import UniformDelay, AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.node import GossipNode
from gossipy.model.handler import LimitedMergeTMH
from gossipy.model.nn import LogisticRegression
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


set_seed(98765)
X, y = load_classification_dataset("spambase", as_tensor=True)
data_handler = ClassificationDataHandler(X, y, test_size=.1)
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False, auto_assign=True)
topology = StaticP2PNetwork(100, to_numpy_array(random_regular_graph(20, 100, seed=42)))
net = LogisticRegression(data_handler.Xtr.shape[1], 2)

nodes = GossipNode.generate(
    data_dispatcher=dispatcher,
    p2p_net=topology,
    model_proto=LimitedMergeTMH(
        net=net,
        optimizer=torch.optim.SGD,
        optimizer_params={
            "lr": 1,
            "weight_decay": .001
        },
        criterion=torch.nn.CrossEntropyLoss(),
    ),
    round_len=100,
    sync=True
)

simulator = GossipSimulator(
    nodes=nodes,
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    delay=UniformDelay(0,10),
    online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=.1, #Simulate the possibility of message dropping,
    sampling_eval=.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=1000)

plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")