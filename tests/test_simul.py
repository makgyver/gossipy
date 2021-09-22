import os
import sys
import torch
import tempfile
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import unittest
from unittest import mock
import pytest

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from gossipy.utils import torch_models_eq
from gossipy.model.nn import TorchMLP
from gossipy.model.handler import ModelHandler, TorchModelHandler
from gossipy.data import DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.node import GossipNode, SimpleTokenAccount
import gossipy.simul as simul
from gossipy.simul import GossipSimulator, TokenizedGossipSimulator, repeat_simulation
from gossipy import AntiEntropyProtocol, CreateModelMode, model, set_seed


def test_GossipSimulator():
    TorchModelHandler._CACHE.clear()
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
    mhpar = {"net" : net,
                "optimizer" : SGD,
                "l2_reg": 0.001,
                "criterion" : CrossEntropyLoss(),
                "learning_rate" : .1,
                "create_model_mode" : CreateModelMode.UPDATE_MERGE}
    gossip = GossipSimulator(data_dispatcher=data_dispatcher,
                             delta=5,
                             protocol=AntiEntropyProtocol.PULL,
                             gossip_node_class=GossipNode,
                             model_handler_class=TorchModelHandler,
                             model_handler_params=mhpar,
                             topology=None,
                             drop_prob=0,
                             online_prob=1,
                             round_synced=True)

    assert gossip.initialized == False
    with pytest.raises(AssertionError):
        gossip.start()
    gossip.init_nodes()
    assert gossip.initialized == True
    assert gossip.data_dispatcher == data_dispatcher
    assert gossip.delta == 5
    assert len(gossip.nodes) == 2
    assert type(gossip.nodes[0]) == GossipNode
    assert gossip.protocol == AntiEntropyProtocol.PULL
    assert gossip.topology is None
    assert gossip.online_prob == 1
    assert gossip.drop_prob == 0
    assert gossip.nodes[0].sync
    assert gossip.gossip_node_class == GossipNode
    assert gossip.model_handler_class == TorchModelHandler
    assert gossip.model_handler_params == mhpar
    
    #assert torch_models_eq(gossip.nodes[0].model_handler.model, net)
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

    eval_list, eval_user_list = repeat_simulation(
        gossip_simulator=gossip,
        n_rounds=10,
        repetitions=2,
        verbose=False
    )

    assert gossip.data_dispatcher == data_dispatcher
    assert gossip.delta == 5
    assert len(gossip.nodes) == 2
    assert type(gossip.nodes[0]) == GossipNode
    assert gossip.protocol == AntiEntropyProtocol.PULL
    assert gossip.topology is None
    assert gossip.online_prob == 1
    assert gossip.drop_prob == 0
    assert gossip.nodes[0].sync


class TestStringMethods(unittest.TestCase):

    def test_fail(self):
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
                                drop_prob=0.9,
                                online_prob=1,
                                round_synced=True)
        gossip.init_nodes()
        with self.assertLogs(logger='gossipy', level='INFO') as cm:
            gossip.start(10)
            assert cm.output[1] != "INFO:gossipy:# Failed messages: 0"

@mock.patch("%s.simul.plt" % __name__)
def test_plot(mock_plt):
    simul.plot_evaluation([[{"acc": 1}]], "test")
    mock_plt.title.assert_called_once_with("test")
    assert mock_plt.figure.called


def test_TokenizedGossipSimulator():
    TorchModelHandler._CACHE.clear()
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
    gossip = TokenizedGossipSimulator(data_dispatcher=data_dispatcher,
    token_account_class=SimpleTokenAccount,
                            token_account_params={"C": 2},
                            utility_fun=lambda mh1, mh2: 1,
                             delta=10,
                             protocol=AntiEntropyProtocol.PUSH,
                             gossip_node_class=GossipNode,
                             model_handler_class=TorchModelHandler,
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
    gossip.init_nodes()
    assert gossip.data_dispatcher == data_dispatcher
    assert gossip.delta == 10
    assert len(gossip.nodes) == 2
    assert type(gossip.nodes[0]) == GossipNode
    assert gossip.protocol == AntiEntropyProtocol.PUSH
    assert gossip.topology is None
    assert gossip.online_prob == 1
    assert gossip.drop_prob == 0
    assert gossip.nodes[0].sync
    
    assert not torch_models_eq(gossip.nodes[0].model_handler.model, net)

    evals, evals_user = gossip.start(100)
    print(ModelHandler._CACHE)
    assert len(evals) == 100
    assert len(evals_user) == 100

    tmp = tempfile.NamedTemporaryFile()
    gossip.save(tmp.name)
    
    gossip2 = TokenizedGossipSimulator.load(tmp.name)
    g2x, g2y = gossip2.data_dispatcher.data_handler.get_eval_set()
    x, y = data_dispatcher.data_handler.get_eval_set()

    assert torch.all(g2x == x)
    assert torch.all(g2y == y)
    assert gossip2.delta == 10
    assert len(gossip2.nodes) == 2
    assert type(gossip2.nodes[0]) == GossipNode
    assert gossip2.protocol == AntiEntropyProtocol.PUSH
    assert gossip2.topology is None
    assert gossip2.online_prob == 1
    assert gossip2.drop_prob == 0
    assert gossip2.nodes[0].sync

    #plot_evaluation([evals], "test")

    gossip.protocol = AntiEntropyProtocol.PULL
    eval_list, eval_user_list = repeat_simulation(
        gossip_simulator=gossip,
        n_rounds=50,
        repetitions=2,
        verbose=False
    )

    assert gossip.data_dispatcher == data_dispatcher
    assert gossip.delta == 10
    assert len(gossip.nodes) == 2
    assert type(gossip.nodes[0]) == GossipNode
    assert gossip.protocol == AntiEntropyProtocol.PULL
    assert gossip.topology is None
    assert gossip.online_prob == 1
    assert gossip.drop_prob == 0
    assert gossip.nodes[0].sync