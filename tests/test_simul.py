from gossipy.utils import torch_models_eq
from gossipy import data
import gossipy
from gossipy.model.nn import TorchMLP
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
import os
import sys
import torch
import tempfile
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from unittest import mock
sys.path.insert(0, os.path.abspath('..'))

from gossipy.model.handler import TorchModelHandler
from gossipy.node import GossipNode
import gossipy.simul as simul
from gossipy.simul import GossipSimulator, repeat_simulation
from gossipy import AntiEntropyProtocol, CreateModelMode, set_seed


def test_GossipSimulator():
    set_seed(42)
    Xtr = torch.FloatTensor([[0,1],[-1,0],[-1,1],
                             [1,-1],[-1,-2],[2,1],
                             [0,2], [-2,2],
                             [0,-2], [2,-2]])
    ytr = torch.LongTensor([0, 0, 0,
                            1, 1, 1,
                            0, 0,
                            1, 1])
    cdh = ClassificationDataHandler(Xtr, ytr, 0.4, 42)
    data_dispatcher = DataDispatcher(cdh, 2, True)
    net = TorchMLP(2, 2, (2,))
    gossip = GossipSimulator(data_dispatcher=data_dispatcher,
                             delta=5,
                             protocol=AntiEntropyProtocol.PULL,
                             gossip_node_class=GossipNode,model_handler_class=TorchModelHandler,
                             model_handler_params={
                                 "net" : net,
                                 "optimizer" : SGD,
                                 "l2_reg": 0.001,
                                 "criterion" : CrossEntropyLoss(),
                                 "learning_rate" : .1,
                                 "create_model_mode" : CreateModelMode.UPDATE_MERGE},
                             topology=None,
                             drop_prob=0,
                             online_prob=1,
                             round_synced=True)

    assert gossip.data_dispatcher == data_dispatcher
    assert gossip.delta == 5
    assert len(gossip.nodes) == 2
    assert type(gossip.nodes[0]) == GossipNode
    assert gossip.protocol == AntiEntropyProtocol.PULL
    assert gossip.topology is None
    assert gossip.online_prob == 1
    assert gossip.drop_prob == 0
    assert gossip.nodes[0].sync
    
    assert torch_models_eq(gossip.nodes[0].model_handler.model, net)
    gossip._init_nodes()
    assert not torch_models_eq(gossip.nodes[0].model_handler.model, net)

    evals, evals_user = gossip.start(10)
    assert len(evals) == 10
    assert len(evals_user) == 10

    tmp = tempfile.NamedTemporaryFile()
    gossip.save(tmp.name)

    gossip2 = GossipSimulator.load(tmp.name)
    g2x, g2y = gossip2.data_dispatcher.data_handler.get_eval_set()
    x, y = data_dispatcher.data_handler.get_eval_set()

    assert torch.all(g2x == x)
    assert torch.all(g2y == y)
    assert gossip2.delta == 5
    assert len(gossip2.nodes) == 2
    assert type(gossip2.nodes[0]) == GossipNode
    assert gossip2.protocol == AntiEntropyProtocol.PULL
    assert gossip2.topology is None
    assert gossip2.online_prob == 1
    assert gossip2.drop_prob == 0
    assert gossip2.nodes[0].sync

    #plot_evaluation([evals], "test")

    sims, eval_list, eval_user_list = repeat_simulation(
        data_dispatcher=data_dispatcher,
        round_delta=5,
        protocol=AntiEntropyProtocol.PULL,
        gossip_node_class=GossipNode,model_handler_class=TorchModelHandler,
        model_handler_params={
            "net" : net,
            "optimizer" : SGD,
            "l2_reg": 0.001,
            "criterion" : CrossEntropyLoss(),
            "learning_rate" : .1,
            "create_model_mode" : CreateModelMode.UPDATE_MERGE},
        topology=None,
        n_rounds=10,
        repetitions=2,
        round_synced=True,
        verbose=False
    )

    assert len(sims) == 2
    assert isinstance(sims[0], GossipSimulator)
    gossip = sims[0]
    assert gossip.data_dispatcher == data_dispatcher
    assert gossip.delta == 5
    assert len(gossip.nodes) == 2
    assert type(gossip.nodes[0]) == GossipNode
    assert gossip.protocol == AntiEntropyProtocol.PULL
    assert gossip.topology is None
    assert gossip.online_prob == 1
    assert gossip.drop_prob == 0
    assert gossip.nodes[0].sync


@mock.patch("%s.simul.plt" % __name__)
def test_plot(mock_plt):
    simul.plot_evaluation([[{"acc": 1}]], "test")
    mock_plt.title.assert_called_once_with("test")
    assert mock_plt.figure.called
