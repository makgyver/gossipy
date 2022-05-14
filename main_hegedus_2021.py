import torch
from torch.nn.modules.loss import CrossEntropyLoss
from networkx import to_numpy_matrix
from networkx.generators.random_graphs import random_regular_graph
from gossipy import set_seed
from gossipy.core import UniformDelay, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode, PartitioningBasedNode, SamplingBasedNode
from gossipy.model.handler import PartitionedTMH, SamplingTMH, TorchModelHandler
from gossipy.model.sampling import TorchModelPartition
from gossipy.model.nn import LogisticRegression
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, TokenizedGossipSimulator, repeat_simulation
from gossipy.flow_control import RandomizedTokenAccount

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
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False)
topology = to_numpy_matrix(random_regular_graph(20, 100, seed=42))
net = LogisticRegression(data_handler.Xtr.shape[1], 2)


simulator = TokenizedGossipSimulator(
#simulator = GossipSimulator(
    data_dispatcher=dispatcher,
    token_account_class=RandomizedTokenAccount, #Coincides with the paper's setting
    token_account_params={"C": 20, "A": 10},
    utility_fun=lambda mh1, mh2: 1, #The utility function is always = 1 (i.e., utility is not used)
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    #gossip_node_class=SamplingBasedNode,
    gossip_node_class=PartitioningBasedNode,
    gossip_node_params={},
    #model_handler_class=SamplingTMH,
    model_handler_class=PartitionedTMH,
    model_handler_params={
        #"sample_size" : .25,
        "tm_partition": TorchModelPartition(net, 4),
        "net" : net,
        "optimizer" : torch.optim.SGD,
        "optimizer_params" : {
            "lr": 1,
            "weight_decay": .001
        },
        "criterion" : CrossEntropyLoss(),
        "create_model_mode" : CreateModelMode.UPDATE #CreateModelMode.MERGE_UPDATE
    },
    topology=topology,
    delay=UniformDelay(0, 10),
    online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
    #drop_prob=.1, #Simulates the possibility of message dropping
    sampling_eval=.1,
    round_synced=True
)

res = repeat_simulation(
    gossip_simulator=simulator,
    n_rounds=1000, #500
    repetitions=1,
    verbose=True
)