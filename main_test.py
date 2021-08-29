import torch
import pandas as pd
import numpy as np
from numpy.random import choice
from typing import OrderedDict, Union
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
#from baseline import sklearn_mlp, torch_mlp
from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode#, UAGossipNode, MABGossipNode
from gossipy.model.handler import TorchModelHandler
from gossipy.model.nn import TorchMLP
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, repeat_simulation
from gossipy.utils import print_flush


set_seed(98765)
X, y = load_classification_dataset("breast", as_tensor=True)
data_handler = ClassificationDataHandler(X, y, test_size=.3)
dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=True)

#set_seed(98765)
res = repeat_simulation(data_dispatcher=dispatcher,
                        round_delta=100,
                        protocol=AntiEntropyProtocol.PUSH, 
                        gossip_node_class=GossipNode,
                        model_handler_class=TorchModelHandler,
                        model_handler_params={"net" : TorchMLP(data_handler.Xtr.shape[1], 2), #TorchMLP
                                              "optimizer" : torch.optim.SGD,
                                              "l2_reg": 0.001,
                                              "criterion" : CrossEntropyLoss(),
                                              "learning_rate" : .1,
                                              "create_model_mode" : CreateModelMode.UPDATE_MERGE},
                        topology_fun=None,
                        n_rounds=100,
                        repetitions=5,
                        round_synced=True,
                        verbose=True)


# mu1 = np.mean(res[1], axis=0)
# mu2 = np.mean(res2[1], axis=0)
# plt.plot(range(len(mu1)), mu1)
# plt.plot(range(len(mu2)), mu2)
# plt.show()
