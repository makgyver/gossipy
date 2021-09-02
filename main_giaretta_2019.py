import networkx as nx
from networkx.generators.trees import random_tree
from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import Pegasos
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import repeat_simulation


set_seed(42)
X, y = load_classification_dataset("spambase", as_tensor=True)
y = 2*y - 1 #convert 0/1 labels to -1/1
data_handler = ClassificationDataHandler(X, y, test_size=.1)

topology = nx.to_numpy_matrix(random_tree(data_handler.size()))

res = repeat_simulation(data_dispatcher=DataDispatcher(data_handler, eval_on_user=False),
                        round_delta=100,
                        protocol=AntiEntropyProtocol.PUSH, 
                        gossip_node_class=GossipNode,
                        model_handler_class=PegasosHandler,
                        model_handler_params={"net" : Pegasos(data_handler.size(1)),
                                              "lam" : .01,
                                              "create_model_mode" : CreateModelMode.MERGE_UPDATE},
                        topology=topology,
                        n_rounds=100,
                        repetitions=1,
                        round_synced=False,
                        verbose=True)

