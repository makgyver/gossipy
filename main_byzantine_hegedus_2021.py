import torch
from torch.nn.modules.loss import CrossEntropyLoss
from gossipy.model.nn import LogisticRegression
from gossipy.utils import plot_evaluation
from gossipy.simul import SimulationReport, TokenizedGossipSimulator
from byzantine_report import ByzantineSimulationReport
from gossipy.data.handler import ClassificationDataHandler
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.model.handler import PartitionedTMH, TorchModelPartition, TorchModel
from gossipy.node import PartitioningBasedNode
from byzantine_generate import generate_nodes
from math import ceil, floor
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformDelay
from gossipy import set_seed
from gossipy.flow_control import RandomizedTokenAccount
from byzantine_handler import BackGradientAttackMixin, AvoidReportMixin, SameValueAttackMixin
from networkx import to_numpy_matrix
from networkx.generators.random_graphs import random_regular_graph
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from scipy.spatial import distance
import numpy as np
from typing import Any, Callable, Tuple, Dict, Optional, Union, Iterable


# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


class PartitionedTMHSameValueAttackHandler(SameValueAttackMixin, PartitionedTMH, AvoidReportMixin):
    def __init__(self,
                 net: TorchModel,
                 tm_partition: TorchModelPartition,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True):
        # BackGradientAttackMixin.__init__(self)
        PartitionedTMH.__init__(self,
                                net,
                                tm_partition,
                                optimizer,
                                optimizer_params,
                                criterion,
                                local_epochs,
                                batch_size,
                                create_model_mode,
                                copy_model)


def eval(data_in, prop):
    if not data_in:
        plus = "+"
    else:
        plus = ""

    set_seed(98765)
    X, y = load_classification_dataset("spambase", as_tensor=True)
    data_handler = ClassificationDataHandler(X, y, test_size=.1)
    dispatcher = DataDispatcher(
        data_handler, n=100, eval_on_user=False, auto_assign=True)

    if data_in:
        total_nb = dispatcher.size()
        malicious_nb = ceil(dispatcher.size() * prop)
        normal_nb = total_nb - malicious_nb
    else:
        total_nb = ceil(dispatcher.size() * (1. + prop))
        malicious_nb = total_nb - dispatcher.size()
        normal_nb = dispatcher.size()

    topology = StaticP2PNetwork(total_nb, to_numpy_matrix(
        random_regular_graph(20, total_nb, seed=42)))
    net = LogisticRegression(data_handler.Xtr.shape[1], 2)

    nodes = generate_nodes(cls=PartitioningBasedNode,
                           data_dispatcher=dispatcher,
                           p2p_net=topology,
                           round_len=100,
                           model_proto=[(normal_nb, PartitionedTMH(
                               net=net,
                               tm_partition=TorchModelPartition(net, 4),
                               optimizer=torch.optim.SGD,
                               optimizer_params={
                                   "lr": 1,
                                   "weight_decay": .001
                               },
                               criterion=CrossEntropyLoss(),
                               create_model_mode=CreateModelMode.UPDATE  # CreateModelMode.MERGE_UPDATE
                           )),
                               (malicious_nb, PartitionedTMHSameValueAttackHandler(
                                   net=net,
                                   tm_partition=TorchModelPartition(net, 4),
                                   optimizer=torch.optim.SGD,
                                   optimizer_params={
                                       "lr": 1,
                                       "weight_decay": .001
                                   },
                                   criterion=CrossEntropyLoss(),
                                   create_model_mode=CreateModelMode.UPDATE  # CreateModelMode.MERGE_UPDATE
                               ), data_in)],
                           sync=True)

    simulator = TokenizedGossipSimulator(
        nodes=nodes,
        data_dispatcher=dispatcher,
        token_account=RandomizedTokenAccount(C=20, A=10),
        # The utility function is always = 1 (i.e., utility is not used)
        utility_fun=lambda mh1, mh2, msg: 1,
        delta=100,
        protocol=AntiEntropyProtocol.PUSH,
        delay=UniformDelay(0, 10),
        # online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
        # drop_prob=.1, #Simulates the possibility of message dropping
        sampling_eval=.1,
    )

    report = ByzantineSimulationReport()
    simulator.add_receiver(report)
    simulator.init_nodes(seed=42)
    simulator.start(n_rounds=1000)

    plot_evaluation([[ev for _, ev in report.get_evaluation(False)]],
                    "Overall test results", "../hegedius2021_{strplus}{nb:.0f}%.png".format(strplus=plus, nb=prop*100.))


for i in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    for val in [False, True]:
        eval(val, i)
