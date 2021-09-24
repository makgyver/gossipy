
from gossipy.model.sampling import TorchModelPartition, TorchModelSampling
from gossipy.utils import torch_models_eq
import os
import sys
import copy
import torch
import numpy as np
from torch.optim import SGD
from torch.nn import functional as F
import pytest

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from gossipy.node import CacheNeighNode, GeneralizedTokenAccount, GossipNode, PassThroughNode, PartitioningBasedNode, PurelyProactiveTokenAccount, PurelyReactiveTokenAccount, RandomizedTokenAccount, SamplingBasedNode, SimpleTokenAccount, TokenAccount
from gossipy.model.handler import PartitionedTMH, PegasosHandler, SamplingTMH, TorchModelHandler
from gossipy.model.nn import Pegasos, TorchMLP
from gossipy import AntiEntropyProtocol, CreateModelMode, MessageType, set_seed

def test_GossipNode():
    TorchModelHandler._CACHE.clear()
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
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, mh.model)
    assert msg.sender == g.idx

    assert g.receive(0, msg) is None

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PULL)
    assert msg.value is None
    assert msg.type == MessageType.PULL
    response = g.receive(0, msg)
    assert response.type ==  MessageType.REPLY
    assert response.sender == 0
    assert response.receiver == 0
    assert torch_models_eq(TorchModelHandler._CACHE[response.value[0]].value.model, mh.model)

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PUSH_PULL)
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, mh.model)
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


def test_PassThroughNode():
    TorchModelHandler._CACHE.clear()
    params = {
        "net" : Pegasos(2),
        "lam" : .1,
        "create_model_mode" : CreateModelMode.UPDATE_MERGE
    }
    mh = PegasosHandler(**params)
    Xtr = torch.FloatTensor([[1,2], [3,4]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])
    Xte = torch.FloatTensor([[1,2], [3,4]])
    yte = torch.FloatTensor([[1, 0], [0, 1]])

    g = PassThroughNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=mh,
        known_nodes=np.array([0,1,1]),
        sync=True
    )
    g.init_model()

    assert g.n_neighs == 2
    assert g.idx == 0
    assert g.data == ((Xtr, ytr), (Xte, yte))
    assert g.n_nodes == 10
    assert g.round_len == 10
    assert 0 <= g.delay < g.round_len
    assert np.all(g.known_nodes == np.array([1,2]))

    assert not g.timed_out(g.delay - 1)
    assert g.timed_out(g.delay)

    g2 = PassThroughNode(
        idx = 1,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=PegasosHandler(**params),
        known_nodes=np.array([0,1,1,1,1,1,1]),
        sync=True
    )
    g2.init_model()

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PUSH)
    assert msg.sender == 0
    assert msg.receiver == 1
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, mh.model)
    assert msg.value[1] == 2
    assert msg.type == MessageType.PUSH

    set_seed(987654)
    assert torch.allclose(g.model_handler.model.model, g2.model_handler.model.model)

    g2.receive(0, msg)

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PUSH)
    g2.receive(0, msg)

    assert torch.allclose(g.model_handler.model.model, g2.model_handler.model.model)

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PUSH_PULL)
    assert msg.sender == 0
    assert msg.receiver == 1
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, mh.model)
    assert msg.value[1] == 2
    assert msg.type == MessageType.PUSH_PULL

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PULL)
    assert msg.sender == 0
    assert msg.receiver == 1
    assert msg.value is None
    assert msg.type == MessageType.PULL

    response = g2.receive(0, msg)
    assert response.type ==  MessageType.REPLY
    assert response.sender == 1
    assert response.receiver == 0
    assert response.value[1] == 6
    assert isinstance(TorchModelHandler._CACHE[response.value[0]].value, PegasosHandler)
    
    with pytest.raises(ValueError):
        g.send(0, 1, protocol=10)


def test_CacheNeighNode():
    TorchModelHandler._CACHE.clear()
    params = {
        "net" : Pegasos(2),
        "lam" : .1,
        "create_model_mode" : CreateModelMode.UPDATE_MERGE
    }
    mh = PegasosHandler(**params)
    Xtr = torch.FloatTensor([[1,2], [3,4]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])
    Xte = torch.FloatTensor([[1,2], [3,4]])
    yte = torch.FloatTensor([[1, 0], [0, 1]])

    g = CacheNeighNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=mh,
        known_nodes=np.array([0,1,1]),
        sync=True
    )
    g.init_model()

    assert isinstance(g.cache, dict)
    assert len(g.cache) == 2
    assert set(g.cache.keys()) == set([1,2])
    assert g.idx == 0
    assert g.data == ((Xtr, ytr), (Xte, yte))
    assert g.n_nodes == 10
    assert g.round_len == 10
    assert 0 <= g.delay < g.round_len
    assert np.all(g.known_nodes == np.array([1,2]))

    assert not g.timed_out(g.delay - 1)
    assert g.timed_out(g.delay)

    g.model_handler._update(g.data[0])

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PUSH)
    assert msg.sender == 0
    assert msg.receiver == 1
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, mh.model)
    assert msg.type == MessageType.PUSH

    set_seed(987654)

    g2 = CacheNeighNode(
        idx = 1,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=PegasosHandler(**params),
        known_nodes=np.array([0,1,1,1,1,1,1]),
        sync=True
    )
    g2.init_model()

    old_model = copy.deepcopy(g2.model_handler.model.model)
    g2.receive(0, msg)

    assert torch.allclose(old_model, g2.model_handler.model.model)
    assert torch.allclose(TorchModelHandler._CACHE[g2.cache[0]].value.model.model, g.model_handler.model.model)

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PUSH_PULL)
    assert msg.sender == 0
    assert msg.receiver == 1
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, mh.model)
    assert msg.type == MessageType.PUSH_PULL

    msg = g.send(0, 1, protocol=AntiEntropyProtocol.PULL)
    assert msg.sender == 0
    assert msg.receiver == 1
    assert msg.value is None
    assert msg.type == MessageType.PULL

    response = g2.receive(0, msg)
    assert response.type ==  MessageType.REPLY
    assert response.sender == 1
    assert response.receiver == 0
    assert isinstance(TorchModelHandler._CACHE[response.value[0]].value, PegasosHandler)
    
    with pytest.raises(ValueError):
        g.send(0, 1, protocol=10)


def test_PBGossipNode():
    TorchModelHandler._CACHE.clear()
    set_seed(987654)
    mlp = TorchMLP(2, 2, (4,))
    part = TorchModelPartition(mlp, 3)
    params = {
        "net" : mlp,
        "tm_partition": part,
        "optimizer" : SGD,
        "l2_reg": 0.001,
        "criterion" : F.mse_loss,
        "learning_rate" : .1,
    }
    tmh = PartitionedTMH(**params)
    tmh.init()
    Xtr = torch.FloatTensor([[1,2], [3,4]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])
    Xte = torch.FloatTensor([[1,2], [3,4]])
    yte = torch.FloatTensor([[1, 0], [0, 1]])

    g = PartitioningBasedNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=tmh,
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
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, tmh.model)
    assert 0 <= msg.value[1] <= 2
    assert msg.sender == g.idx

    assert g.receive(0, msg) is None

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PULL)
    assert msg.value is None
    assert msg.type == MessageType.PULL
    response = g.receive(0, msg)
    assert response.type ==  MessageType.REPLY
    assert response.sender == 0
    assert response.receiver == 0
    assert torch_models_eq(TorchModelHandler._CACHE[response.value[0]].value.model, tmh.model)
    assert 0 <= response.value[1] <= 2

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PUSH_PULL)
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, tmh.model)
    assert 0 <= msg.value[1] <= 2
    assert msg.type == MessageType.PUSH_PULL

    with pytest.raises(ValueError):
        g.send(0, 0, protocol=10)
    
    res = g.evaluate()
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.25
    assert res["f1_score"] == 1/3
    assert res["auc"] == 0.
    assert res["recall"] == 0.5

    res = g.evaluate((Xte, yte))
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.25
    assert res["f1_score"] == 1/3
    assert res["auc"] == 0.
    assert res["recall"] == 0.5

    g = PartitioningBasedNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=tmh,
        known_nodes=np.identity(10)[0],
        sync=False
    )
    g.init_model()
    assert g.timed_out(g.delay)
    assert g.known_nodes == np.array([0])
    assert g.get_peer() == 0


def test_SBGossipNode():
    TorchModelHandler._CACHE.clear()
    set_seed(987654)
    mlp = TorchMLP(2, 2, (4,))
    params = {
        "sample_size": .3,
        "net" : mlp,
        "optimizer" : SGD,
        "l2_reg": 0.001,
        "criterion" : F.mse_loss,
        "learning_rate" : .1,
    }
    tmh = SamplingTMH(**params)
    tmh.init()
    Xtr = torch.FloatTensor([[1,2], [3,4]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])
    Xte = torch.FloatTensor([[1,2], [3,4]])
    yte = torch.FloatTensor([[1, 0], [0, 1]])

    g = SamplingBasedNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=tmh,
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
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, tmh.model)
    assert msg.value[1] == .3
    assert msg.sender == g.idx

    assert g.receive(0, msg) is None

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PULL)
    assert msg.value is None
    assert msg.type == MessageType.PULL
    response = g.receive(0, msg)
    assert response.type ==  MessageType.REPLY
    assert response.sender == 0
    assert response.receiver == 0
    assert torch_models_eq(TorchModelHandler._CACHE[response.value[0]].value.model, tmh.model)
    assert response.value[1] == .3

    msg = g.send(0, 0, protocol=AntiEntropyProtocol.PUSH_PULL)
    assert torch_models_eq(TorchModelHandler._CACHE[msg.value[0]].value.model, tmh.model)
    assert msg.value[1] == .3
    assert msg.type == MessageType.PUSH_PULL

    with pytest.raises(ValueError):
        g.send(0, 0, protocol=10)
    
    res = g.evaluate()
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.25
    assert res["f1_score"] == 1/3
    assert res["auc"] == 1
    assert res["recall"] == .5

    res = g.evaluate((Xte, yte))
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.25
    assert res["f1_score"] == 1/3
    assert res["auc"] == 1
    assert res["recall"] == .5

    g = PartitioningBasedNode(
        idx = 0,
        data = ((Xtr, ytr), (Xte, yte)),
        round_len=10,
        n_nodes=10,
        model_handler=tmh,
        known_nodes=np.identity(10)[0],
        sync=False
    )
    g.init_model()
    assert g.timed_out(g.delay)
    assert g.known_nodes == np.array([0])
    assert g.get_peer() == 0


def test_TokenAccount():
    ta = TokenAccount()
    assert ta.n_tokens == 0
    ta.add(3)
    assert ta.n_tokens == 3
    ta.sub()
    assert ta.n_tokens == 2
    with pytest.raises(NotImplementedError):
        ta.proactive()
    with pytest.raises(NotImplementedError):
        ta.reactive(1)

def test_PurelyTA():
    ppta = PurelyProactiveTokenAccount()
    assert ppta.n_tokens == 0
    assert ppta.proactive() == 1
    assert ppta.reactive(1) == 0
    assert ppta.reactive(10) == 0

    prta = PurelyReactiveTokenAccount(10)
    assert prta.k == 10
    assert prta.proactive() == 0
    assert prta.reactive(1) == 10
    assert prta.reactive(10) == 100


def test_SimpleTA():
    with pytest.raises(AssertionError):
        SimpleTokenAccount(C=0)
    sta = SimpleTokenAccount(C=1)
    assert sta.capacity == 1
    assert not sta.proactive()
    assert not sta.proactive()
    sta.add()
    assert sta.proactive()
    assert sta.reactive(1) == 1
    assert sta.reactive(0) == 1


def test_GeneralizedTA():
    with pytest.raises(AssertionError):
        GeneralizedTokenAccount(C=1, A=0)
    
    with pytest.raises(AssertionError):
        GeneralizedTokenAccount(C=3, A=4)

    gta = GeneralizedTokenAccount(C=1, A=1)
    assert gta.capacity == 1
    assert gta.reactivity == 1
    assert not gta.proactive()
    assert gta.reactive(1) == 0
    assert gta.reactive(0) == 0
    gta.add()
    assert gta.proactive()
    assert gta.reactive(0) == 0
    assert gta.reactive(10) == 1


def test_RandomizedTA():
    with pytest.raises(AssertionError):
        RandomizedTokenAccount(C=1, A=0)
    
    with pytest.raises(AssertionError):
        RandomizedTokenAccount(C=3, A=4)

    rta = RandomizedTokenAccount(C=2, A=2)
    assert rta.capacity == 2
    assert rta.reactivity == 2
    assert rta.proactive() == 0
    assert rta.reactive(1) == 0
    assert rta.reactive(0) == 0
    rta.add()
    assert rta.proactive() == 0
    assert rta.reactive(0) == 0
    assert rta.reactive(10) == 1
    rta.add()
    assert rta.proactive() == 1
    assert rta.reactive(0) == 0
    assert rta.reactive(10) == 1
    rta.add(10)
    assert rta.proactive() == 1
    assert rta.reactive(0) == 0
    assert rta.reactive(10) == 6

    rta = RandomizedTokenAccount(C=5, A=1)
    assert rta.proactive() == 0
    assert rta.reactive(1) == 0
    assert rta.reactive(0) == 0
    rta.add(2)
    assert rta.proactive() == 0.4
    assert rta.reactive(1) == 2
    assert rta.reactive(0) == 0