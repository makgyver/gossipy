from gossipy.utils import torch_models_eq
import os
import sys
import torch
from torch.optim import SGD
from torch.nn import functional as F
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from gossipy.model import TorchModel
from gossipy import CreateModelMode, set_seed
from gossipy.model.handler import AdaLineHandler, ModelHandler, PartitionedTMH, PegasosHandler, SamplingTMH, TorchModelHandler
from gossipy.model.nn import AdaLine, Pegasos, TorchMLP, TorchPerceptron
from gossipy.model.sampling import TorchModelPartition, TorchModelSampling

def test_TorchModel():
    tm = TorchModel()
    with pytest.raises(NotImplementedError):
        tm.init_weights()
    
    assert tm.get_size() == 0
    assert str(tm) == "TorchModel(size=0)"
    assert len(tm.get_params_list()) == 0

    class TempModel(TorchModel):
        def __init__(self):
            super(TempModel, self).__init__()
            self.w = torch.nn.Linear(2, 1) # 2 + bias
    
    tm = TempModel()
    assert tm.get_size() == 3

def test_TorchPerceptron():
    p = TorchPerceptron(2)
    p.init_weights()
    X = torch.FloatTensor([[0, 1]])
    y = p(X)
    assert y.shape == torch.Size([1, 1])
    s1 = """Sequential(
  (linear): Linear(in_features=2, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)"""
    s = "TorchPerceptron(size=3)\n" + s1
    assert str(p) == s

def test_TorchMLP():
    mlp = TorchMLP(2, 2, (4,))
    mlp.init_weights()
    X = torch.FloatTensor([[0, 1]])
    y = mlp(X)
    assert y.shape == torch.Size([1,2])

    s1 = """Sequential(
  (linear_1): Linear(in_features=2, out_features=4, bias=True)
  (activ_1): ReLU()
  (linear_2): Linear(in_features=4, out_features=2, bias=True)
)"""
    s = "TorchMLP(size=22)\n" + s1
    assert str(mlp) == s


def test_ModelHandler():
    mh = ModelHandler()
    assert mh.model is None
    assert mh.mode == CreateModelMode.UPDATE
    assert mh.n_updates == 0

    with pytest.raises(NotImplementedError):
        mh.init(None)
    
    with pytest.raises(NotImplementedError):
        mh._update(None)
    
    with pytest.raises(NotImplementedError):
        mh._merge(None)

    with pytest.raises(NotImplementedError):
        mh.evaluate()
    
    assert mh == mh.copy()
    assert mh.get_size() == 0

def test_TorchModelHandler():
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
    tmh = TorchModelHandler(**params)
    tmh.init()

    assert str(tmh.model) == str(mlp)
    assert tmh.n_updates == 0
    assert tmh.optimizer.param_groups[0]['lr'] == .1
    assert tmh.optimizer.param_groups[0]['weight_decay'] == .001
    Xtr = torch.FloatTensor([[1, 1], [0.1, 0.1]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])

    tmh2 = TorchModelHandler(**params)
    assert not torch_models_eq(tmh.model, tmh2.model)
    tmh2.init()

    tmh(tmh2, (Xtr, ytr))
    assert tmh.n_updates == 1
    assert tmh2.n_updates == 1

    params["create_model_mode"] = CreateModelMode.MERGE_UPDATE
    tmh = TorchModelHandler(**params)
    tmh.init()
    
    tmh(tmh2, (Xtr, ytr))
    assert tmh.n_updates == 2
    assert tmh2.n_updates == 1

    params["create_model_mode"] = CreateModelMode.UPDATE
    tmh = TorchModelHandler(**params)
    tmh.init()

    tmh(tmh2, (Xtr, ytr))
    assert tmh.n_updates == 2
    
    params["create_model_mode"] = 10
    set_seed(987654)
    tmh = TorchModelHandler(**params)
    tmh.init()

    with pytest.raises(ValueError):
        tmh(tmh2, (Xtr, ytr))
    
    #assert not (tmh == "string")
    tmh2 = tmh.copy()
    assert torch_models_eq(tmh.model, tmh2.model)
    assert tmh2 != tmh
    assert not (tmh2 == tmh)

    result = tmh.evaluate((Xtr, ytr))
    assert result["accuracy"] == 0.5
    assert result["recall"] == 1.
    assert result["f1_score"] == 2/3
    assert result["precision"] == .5
    assert result["auc"] == 1.

def test_AdaLine():
    ada = AdaLine(4)
    assert ada.input_dim == 4
    assert torch.all(ada.model == torch.zeros_like(ada.model))

    ada.init_weights()
    assert torch.all(ada.model == torch.zeros_like(ada.model))

    assert torch.all(ada(torch.FloatTensor([1,1,1,1])) == torch.zeros_like(ada.model))

    assert str(ada) == "AdaLine(size=4)"

    peg = Pegasos(4)
    assert str(peg) == "Pegasos(size=4)"


def test_AdaLineHandler():
    ada = AdaLineHandler(AdaLine(2), 0.1, copy_model=False)
    ada.init()

    assert str(ada.model) == str(AdaLine(2))
    assert ada.n_updates == 0
    assert ada.learning_rate == 0.1
    assert ada.mode == CreateModelMode.UPDATE

    X = torch.FloatTensor([[1,1], [1,0]])
    y = torch.LongTensor([1, -1])
    ada._update((X, y))
    assert torch.allclose(ada.model.model, torch.FloatTensor([-0.0100, 0.1000]))

    ada._merge(AdaLineHandler(AdaLine(2), 0.1, copy_model=False))
    assert torch.allclose(ada.model.model, torch.FloatTensor([-0.005, 0.05]))

    ada._update((X, y))
    ada._update((X, y))
    ada._update((X, y))
    res = ada.evaluate((X, y))
    assert res["accuracy"] == res["recall"] == res["f1_score"] == res["auc"] == 1


def test_PegasosHandler():
    pegh = PegasosHandler(Pegasos(2), 0.1, copy_model=False)
    pegh.init()

    assert str(pegh.model) == str(Pegasos(2))
    assert pegh.n_updates == 0
    assert pegh.learning_rate == 0.1
    assert pegh.mode == CreateModelMode.UPDATE

    X = torch.FloatTensor([[1,1], [1,0]])
    y = torch.LongTensor([1, -1])
    pegh._update((X, y))
    assert torch.allclose(pegh.model.model, torch.FloatTensor([0, 5]))

    pegh._merge(PegasosHandler(Pegasos(2), 0.1, copy_model=False))
    assert torch.allclose(pegh.model.model, torch.FloatTensor([0, 2.5]))

    pegh._update((X, y))
    pegh._update((X, y))
    pegh._update((X, y))
    res = pegh.evaluate((X, y))
    assert res["accuracy"] == res["recall"] == res["f1_score"] == res["auc"] == 1


def test_PTMH():
    set_seed(987654)
    mlp = TorchMLP(2, 2, (4,))
    part = TorchModelPartition(mlp, 4)
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

    assert str(tmh.model) == str(mlp)
    assert np.all(tmh.n_updates == [0,0,0,0])
    assert tmh.optimizer.param_groups[0]['lr'] == .1
    assert tmh.optimizer.param_groups[0]['weight_decay'] == .001
    Xtr = torch.FloatTensor([[1, 1], [0.1, 0.1]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])

    tmh2 = PartitionedTMH(**params)
    assert not torch_models_eq(tmh.model, tmh2.model)
    tmh2.init()

    tmh(tmh2, (Xtr, ytr), 0)
    assert np.all(tmh.n_updates == [1,1,1,1])
    assert np.all(tmh2.n_updates == [0,0,0,0])

    #tmh = PartitionedTMH(**params)
    #tmh.init()
    
    tmh(tmh2, (Xtr, ytr), 1)
    assert np.all(tmh.n_updates == [2,2,2,2])
    assert np.all(tmh2.n_updates == [0,0,0,0])

    tmh2(tmh, (Xtr, ytr), 2)
    assert np.all(tmh2.n_updates == [1,1,3,1])
    
    tmh = PartitionedTMH(**params)
    tmh.init()

    tmh2 = tmh.copy()
    assert torch_models_eq(tmh.model, tmh2.model)
    assert tmh2 != tmh
    assert not (tmh2 == tmh)

    result = tmh.evaluate((Xtr, ytr))
    assert result["accuracy"] == 0.5
    assert result["recall"] == 1.
    assert result["f1_score"] == 2/3
    assert result["precision"] == .5
    assert result["auc"] == 1.


def test_STMH():
    set_seed(987654)
    mlp = TorchMLP(2, 2, (4,))
    params = {
        "net" : mlp,
        "optimizer" : SGD,
        "l2_reg": 0.001,
        "criterion" : F.mse_loss,
        "learning_rate" : .1,
    }
    tmh = SamplingTMH(**params)
    tmh.init()

    assert str(tmh.model) == str(mlp)
    assert tmh.n_updates == 0
    assert tmh.optimizer.param_groups[0]['lr'] == .1
    assert tmh.optimizer.param_groups[0]['weight_decay'] == .001
    Xtr = torch.FloatTensor([[1, 1], [0.1, 0.1]])
    ytr = torch.FloatTensor([[1, 0], [0, 1]])

    tmh2 = SamplingTMH(**params)
    assert not torch_models_eq(tmh.model, tmh2.model)
    tmh2.init()

    tmh(tmh2, (Xtr, ytr), TorchModelSampling.sample(.1, mlp))

    result = tmh.evaluate((Xtr, ytr))
    assert result["accuracy"] == 0.5
    assert result["recall"] == 1.
    assert result["f1_score"] == 2/3
    assert result["precision"] == .5
    assert result["auc"] == 0.