
from gossipy.utils import torch_models_eq
import os
import sys
import torch
import numpy as np
from torch.optim import SGD
from torch.nn import functional as F
import pytest

sys.path.insert(0, os.path.abspath('..'))
from gossipy.node import GossipNode
from gossipy.model.handler import TorchModelHandler
from gossipy.model.nn import TorchMLP
from gossipy import AntiEntropyProtocol, CreateModelMode, MessageType, set_seed

def test_GossipNode():
    set_seed(987654)
    mlp = TorchMLP(2, 2, (4,))
    params = {
        "net" : mlp,
        "optimizer" : SGD,
        "l2_reg": 0.001,
        "criterion" : F.mse_loss,
        "learning_rate" : .1,
        "create_model_mode" : CreateModelMode.UPDATE_MERGE
    }
    mh = TorchModelHandler(**params)
    Xtr = torch.FloatTensor([[1,2], [3,4]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])
    Xte = torch.FloatTensor([[1,2], [3,4]])
    yte = torch.FloatTensor([[1, 0], [0, 1]])

    g = GossipNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=mh,
        known_nodes=None,
        sync=True
    )
    g.init_model()

    assert g.idx == 0
    assert g.data == ((Xtr, ytr), (Xte, yte))
    assert g.n_nodes == 10
    assert g.round_len == 10
    assert 0 <= g.delay < g.round_len
    assert g.known_nodes is None

    assert not g.timed_out(g.delay - 1)
    assert g.timed_out(g.delay)

    #not really necessary
    for _ in range(1000):
        assert g.get_peer() != g.idx
    
    assert g.has_test()

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PUSH)
    assert msg.type == MessageType.PUSH
    assert torch_models_eq(msg.value.model, mh.model)
    assert msg.sender == g.idx

    assert g.receive(msg) is None

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PULL)
    assert msg.value is None
    assert msg.type == MessageType.PULL
    response = g.receive(msg)
    assert response.type ==  MessageType.REPLY
    assert response.sender == 0
    assert response.receiver == 0
    assert torch_models_eq(response.value.model, mh.model)

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PUSH_PULL)
    assert torch_models_eq(msg.value.model, mh.model)
    assert msg.type == MessageType.PUSH_PULL

    with pytest.raises(ValueError):
        g.send(0, 0, protocol=10)
    
    res = g.evaluate()
    assert res["accuracy"] == 0.
    assert res["precision"] == 0.
    assert res["f1_score"] == 0.
    assert res["auc"] == 0.
    assert res["recall"] == 0.

    res = g.evaluate((Xte, yte))
    assert res["accuracy"] == 0.
    assert res["precision"] == 0.
    assert res["f1_score"] == 0.
    assert res["auc"] == 0.
    assert res["recall"] == 0.

    g = GossipNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=mh,
        known_nodes=np.identity(10)[0],
        sync=False
    )
    g.init_model()
    assert g.timed_out(g.delay)
    assert g.known_nodes == np.array([0])
    assert g.get_peer() == 0