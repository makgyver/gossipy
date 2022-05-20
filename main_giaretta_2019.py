import networkx as nx
from networkx.generators.trees import random_tree
from gossipy import set_seed
from gossipy import data
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import Pegasos
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, repeat_simulation

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

data_dispatcher = DataDispatcher(data_handler, eval_on_user=False)
data_dispatcher.assign()

topology = StaticP2PNetwork(data_dispatcher.size())

nodes = GossipNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=topology,
    model_proto=PegasosHandler(
        net=Pegasos(data_handler.size(1)),
        lam=.01,
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

res = repeat_simulation(
    gossip_simulator=simulator,
    n_rounds=100,
    repetitions=1, #set values > 1
    verbose=True
)
