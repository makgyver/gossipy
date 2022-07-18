from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformDelay
from gossipy.node import GossipNode
from gossipy.model.handler import TorchModelHandler
from gossipy.model.nn import TorchMLP
from gossipy.data import get_MNIST, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator
from gossipy.utils import plot_evaluation
from typing import Tuple, Dict, Any, Callable
import torch
from byzantine_handler import TorchModelBackGradientAttackHandler, TorchModelRandomModelAttackHandler, TorchModelRandomFullModelAttackHandler, TorchModelSameValueAttackHandler, TorchModelGradientScalingAttackHandler
from byzantine_report import ByzantineSimulationReport
from byzantine_generate import generate_nodes, GenerationType
from byzantine_network import multi_clique_network
from gossipy.model.nn import TorchModel
from math import ceil
import pickle


import copy

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"


class PerceptronMNISTModel(torch.nn.Module):
    def __init__(self):
        super(PerceptronMNISTModel, self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 6, 5)
        # self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.lin1 = torch.nn.Linear(28*28, 100)
        self.lin2 = torch.nn.Linear(100, 10)
        # self.lin3 = torch.nn.Linear(84, 10)
        self.max = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.lin2(self.relu(self.lin1(x))))
        # return self.lin1(self.relu(self.conv1(x.view(x.shape[0], 1, 28, 28))).view(x.shape[0], 24*24*6))
        # return self.lin3(self.relu(self.lin2(self.relu(self.lin1(self.max(
        #     self.relu(self.conv2(self.max(self.relu(self.conv1(x.view(x.shape[0], 1, 28, 28))))))).view(x.shape[0], 256))))))


class TorchPerceptronMNIST(TorchModel):
    def __init__(self):
        super(TorchPerceptronMNIST, self).__init__()
        self.model = PerceptronMNISTModel()

    # docstr-coverage:inherited
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # docstr-coverage:inherited
    def init_weights(self) -> None:
        # torch.nn.init.xavier_uniform_(self.model.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.model.conv2.weight)
        torch.nn.init.xavier_uniform_(self.model.lin1.weight)
        torch.nn.init.xavier_uniform_(self.model.lin2.weight)
        # torch.nn.init.xavier_uniform_(self.model.lin3.weight)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size=%d)\n%s" % (self.get_size(), str(self.model))


class ConvolMNISTModel(torch.nn.Module):
    def __init__(self):
        super(ConvolMNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.lin1 = torch.nn.Linear(20*20*16, 100)
        # self.lin2 = torch.nn.Linear(100, 10)
        # self.lin3 = torch.nn.Linear(84, 10)
        self.max = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.lin1(self.relu(self.conv2(self.relu(self.conv1(x.view(x.shape[0], 1, 28, 28))))).view(x.shape[0], 20*20*16))
        # return self.lin3(self.relu(self.lin2(self.relu(self.lin1(self.max(
        #     self.relu(self.conv2(self.max(self.relu(self.conv1(x.view(x.shape[0], 1, 28, 28))))))).view(x.shape[0], 256))))))


class TorchConvolMNIST(TorchModel):
    def __init__(self):
        super(TorchConvolMNIST, self).__init__()
        self.model = ConvolMNISTModel()

    # docstr-coverage:inherited
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # docstr-coverage:inherited
    def init_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.model.conv1.weight)
        torch.nn.init.xavier_uniform_(self.model.conv2.weight)
        torch.nn.init.xavier_uniform_(self.model.lin1.weight)
        # torch.nn.init.xavier_uniform_(self.model.lin2.weight)
        # torch.nn.init.xavier_uniform_(self.model.lin3.weight)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size=%d)\n%s" % (self.get_size(), str(self.model))


def eval(data_in, prop):
    if not data_in:
        plus = "+"
    else:
        plus = ""

    set_seed(42)
    tr, te = get_MNIST()

    data_handler = ClassificationDataHandler(
        tr[0], tr[1], te[0], te[1], on_device=True)  # X[1], y[1]
    data_dispatcher = DataDispatcher(
        data_handler, n=500, eval_on_user=False, auto_assign=True)  # Normalement n=1000

    if data_in:
        total_nb = data_dispatcher.size()
        malicious_nb = ceil(data_dispatcher.size() * prop)
        normal_nb = total_nb - malicious_nb
    else:
        total_nb = ceil(data_dispatcher.size() * (1. + prop))
        malicious_nb = total_nb - data_dispatcher.size()
        normal_nb = data_dispatcher.size()

    topology = StaticP2PNetwork(total_nb, None)
    #topology = StaticP2PNetwork(total_nb, multi_clique_network(total_nb, 20))
    net = TorchMLP(28*28, 10)
    #net = TorchConvolMNIST()
    model_handler = TorchModelHandler(net=net, optimizer=torch.optim.SGD,
                                      optimizer_params={
                                          "lr": 0.1
                                      }, criterion=torch.nn.CrossEntropyLoss(), on_device=True)
    model_handler_malicious = TorchModelSameValueAttackHandler(net=net, optimizer=torch.optim.SGD,
                                                               optimizer_params={
                                                                   "lr": 0.1,
                                                               }, criterion=torch.nn.CrossEntropyLoss(), on_device=True)

    # For loop to repeat the simulation
    nodes = generate_nodes(GossipNode, data_dispatcher=data_dispatcher,
                           p2p_net=topology,
                           model_proto=((normal_nb, model_handler),
                                        (malicious_nb, model_handler_malicious, data_in)),
                           round_len=100,
                           sync=False,
                           generation_type=GenerationType.NORMAL)

    simulator = GossipSimulator(
        nodes=nodes,
        data_dispatcher=data_dispatcher,
        delta=100,
        protocol=AntiEntropyProtocol.PUSH,
        # delay=UniformDelay(0, 10),
        # online_prob=.2,  # Approximates the average online rate of the STUNner's smartphone traces
        # drop_prob=.1,  # Simulate the possibility of message dropping,
        sampling_eval=0.1
    )

    report = ByzantineSimulationReport()
    simulator.add_receiver(report)
    simulator.init_nodes(seed=42)
    simulator.start(n_rounds=250)

    with open("../mnist-{strplus}{propo:.1f}%-backgradient-convolutif-dizaine.pickle".format(strplus=plus, propo=100*prop), "wb") as file:
        pickle.dump([[ev for _, ev in report.get_evaluation(False)]], file)

    plot_evaluation([[ev for _, ev in report.get_evaluation(False)]],
                    "Overall test results", "../mnist-{strplus}{propo:.1f}%-backgradient-convolutif-dizaine.png".format(strplus=plus, propo=100*prop))
    # plot_evaluation([[ev for _, ev in report.get_evaluation(True)]], "User-wise test results")


eval(True, 0.1)
# for val in [True]:
# for i in [0.00, 0.01, 0.02, 0.05, 0.1, 0.2, 0.15, 0.3, 0.4, 0.5]:  #
# eval(val, i)
#
