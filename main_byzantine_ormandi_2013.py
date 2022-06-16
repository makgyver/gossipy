from gossipy import set_seed
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformDelay
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import AdaLine
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator
from gossipy.utils import plot_evaluation
from typing import Tuple
import torch
from byzantine_handler import AvoidReportMixin, SameValueAttackMixin
from byzantine_report import ByzantineSimulationReport
from byzantine_generate import generate_nodes
from math import ceil
import pickle

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


class PegasosSameValueAttackHandler(SameValueAttackMixin, PegasosHandler):
    pass


class PegasosGradientScalingAttackHandler(AvoidReportMixin, PegasosHandler):
    def __init__(self,
                 scale: float,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True):
        super(PegasosGradientScalingAttackHandler, self).__init__(
            net, learning_rate, create_model_mode, copy_model)
        self.scale = scale

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        for i in range(len(y)):
            self.n_updates += 1
            lr = 1. / (self.n_updates * self.learning_rate)
            y_pred = self.model(x[i:i+1])
            # self.model.model *= (1. - lr * self.learning_rate)
            # self.model.model += ((y_pred * y[i] - 1)
            #                     < 0).float() * (lr * y[i] * x[i])
            self.model.model += self.scale * (((y_pred * y[i] - 1) < 0).float() * (
                lr * y[i] * x[i]) - lr * self.learning_rate * self.model.model)


def eval(data_in, prop):
    if not data_in:
        plus = "+"
    else:
        plus = ""

    set_seed(42)
    X, y = load_classification_dataset("spambase", as_tensor=True)
    y = 2*y - 1  # convert 0/1 labels to -1/1

    data_handler = ClassificationDataHandler(X, y, test_size=.1)
    data_dispatcher = DataDispatcher(
        data_handler, eval_on_user=False, auto_assign=True)

    if data_in:
        total_nb = data_dispatcher.size()
        malicious_nb = ceil(data_dispatcher.size() * prop)
        normal_nb = total_nb - malicious_nb
    else:
        total_nb = ceil(data_dispatcher.size() * (1. + prop))
        malicious_nb = total_nb - data_dispatcher.size()
        normal_nb = data_dispatcher.size()

    topology = StaticP2PNetwork(total_nb, None)
    model_handler = PegasosHandler(net=AdaLine(data_handler.size(1)),
                                   learning_rate=.01,
                                   create_model_mode=CreateModelMode.MERGE_UPDATE)
    model_handler_malicious = PegasosGradientScalingAttackHandler(scale=-1.0, net=AdaLine(data_handler.size(1)),
                                                                  learning_rate=.01,
                                                                  create_model_mode=CreateModelMode.MERGE_UPDATE)

    # For loop to repeat the simulation
    nodes = generate_nodes(GossipNode, data_dispatcher=data_dispatcher,
                           p2p_net=topology,
                           model_proto=((normal_nb, model_handler),
                                        (malicious_nb, model_handler_malicious, True)),
                           round_len=100,
                           sync=False)

    simulator = GossipSimulator(
        nodes=nodes,
        data_dispatcher=data_dispatcher,
        delta=100,
        protocol=AntiEntropyProtocol.PUSH,
        #    delay=UniformDelay(0, 10),
        #    online_prob=.2,  # Approximates the average online rate of the STUNner's smartphone traces
        #    drop_prob=.1,  # Simulate the possibility of message dropping,
        sampling_eval=.1
    )

    report = ByzantineSimulationReport()
    simulator.add_receiver(report)
    simulator.init_nodes(seed=42)
    simulator.start(n_rounds=100)

    with open("../ormandi-{strplus}{propo:.0f}%-backgradient.pickle".format(strplus=plus, propo=100*prop), "wb") as file:
        pickle.dump([[ev for _, ev in report.get_evaluation(False)]], file)

    plot_evaluation([[ev for _, ev in report.get_evaluation(False)]],
                    "Overall test results", "../ormandi-{strplus}{propo:.0f}%-backgradient.png".format(strplus=plus, propo=100*prop))
    #plot_evaluation([[ev for _, ev in report.get_evaluation(True)]], "User-wise test results")


for i in [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    eval(True, i)
