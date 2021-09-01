import os
import sys
from math import isclose
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath('..'))

from gossipy.data import DataDispatcher, DataHandler, load_classification_dataset
from gossipy.data.handler import ClassificationDataHandler



def test_DataHandler():
    dh = DataHandler()

    with pytest.raises(NotImplementedError):
        dh[0]
    
    with pytest.raises(NotImplementedError):
        dh.size()
    
    with pytest.raises(NotImplementedError):
        dh.get_eval_set()
    
    with pytest.raises(NotImplementedError):
        dh.eval_size()
    
    with pytest.raises(NotImplementedError):
        dh.train_size()

def test_ClassificationDataHandler():
    X = torch.FloatTensor([[1,2], [3,4]])
    y = torch.LongTensor([[0], [1]])
    cdh = ClassificationDataHandler(X, y, 0.5, 42)

    assert torch.all(cdh.Xtr == torch.FloatTensor([[1,2]]))
    assert torch.all(cdh.Xte == torch.FloatTensor([[3,4]]))
    assert torch.all(cdh.ytr == torch.LongTensor([0]))
    assert torch.all(cdh.yte == torch.LongTensor([1]))

    cdh = ClassificationDataHandler(X, y, 0, 42)
    assert torch.all(cdh.Xtr == X)
    assert torch.all(cdh.ytr == y)

    X = np.array([[1,2], [3,4]])
    y = np.array([0, 1])
    cdh = ClassificationDataHandler(X, y, 0.5, 42)

    assert np.all(cdh.Xtr == np.array([[1,2]]))
    assert np.all(cdh.Xte == np.array([[3,4]]))
    assert np.all(cdh.ytr == np.array([0]))
    assert np.all(cdh.yte == np.array([1]))

    X, y = cdh[0]
    assert np.all(X == np.array([[1,2]]))
    assert np.all(y == np.array([0]))

    X, y = cdh.at(0)
    assert np.all(X == np.array([[1,2]]))
    assert np.all(y == np.array([0]))

    X, y = cdh.at(0, eval_set=True)
    assert np.all(X == np.array([[3,4]]))
    assert np.all(y == np.array([1]))

    assert cdh.at([], eval_set=True) is None

    assert cdh.size() == 1
    assert cdh.eval_size() == 1
    X, y = cdh.get_eval_set()
    assert np.all(X == np.array([[3,4]]))
    assert np.all(y == np.array([1]))


def test_DataDispatcher():
    X = torch.FloatTensor([[1,2], [3,4], [5,6], [7,8]])
    y = torch.LongTensor([[0], [1], [0], [1]])
    cdh = ClassificationDataHandler(X, y, 0.5, 42)

    dd = DataDispatcher(cdh, 2, True)
    assert dd.data_handler == cdh
    assert dd.n == 2
    assert len(dd.tr_assignments[0]) == 1
    assert len(dd.tr_assignments[1]) == 1
    assert len(dd.te_assignments[0]) == 1
    assert len(dd.te_assignments[1]) == 1
    #TODO: check if the assignments are correct

    (Xtr, ytr), (Xte, yte) = dd[0]
    assert Xtr.shape ==  torch.Size([1, 2])
    assert ytr.shape ==  torch.Size([1, 1])
    assert Xte.shape ==  torch.Size([1, 2])
    assert yte.shape ==  torch.Size([1, 1])

    assert dd.size() == 2
    assert dd.get_eval_set() == cdh.get_eval_set()
    assert dd.has_test()

def test_load_classification_dataset():
    X, y = load_classification_dataset("iris", normalize=False, as_tensor=False)
    assert type(X) == np.ndarray
    assert type(y) == np.ndarray
    assert y.shape[0] == X.shape[0]
    assert len(set(y)) == 3
    assert X.shape[1] == 4

    X, y = load_classification_dataset("breast", normalize=True, as_tensor=True)
    assert type(X) == torch.Tensor
    assert type(y) == torch.Tensor
    assert y.shape[0] == X.shape[0]
    assert len(set(y.numpy().flatten())) == 2
    assert X.shape[1] == 30
    for i in range(30):
        assert isclose(torch.mean(X[:, i]).item(), 0, rel_tol=1e-8, abs_tol=1e-7)
    for i in range(30):
        assert isclose(torch.std(X[:, i]).item(), 1, rel_tol=1e-4, abs_tol=1e-3) #ugly but there is no other way
    
    with pytest.raises(ValueError):
        load_classification_dataset("invalid")

    #TODO: test one of these {"sonar", "ionosphere", "abalone", "banknote", "diabetes"}

    X, y = load_classification_dataset("spambase", normalize=True, as_tensor=True)
    assert type(X) == torch.Tensor
    assert type(y) == torch.Tensor
    assert y.shape[0] == X.shape[0]
    assert y.shape[0] == 4601


