import os
from typing import Any, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import torch
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler


__all__ = ["DataHandler", "DataDispatcher", "load_classification_dataset", "load_recsys_dataset"]


#TODO: get training set?
class DataHandler():
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError()

    def size(self) -> int:
        raise NotImplementedError()

    def get_eval_set(self) -> Tuple[Any, Any]:
        raise NotImplementedError()
    
    def eval_size(self) -> int:
        raise NotImplementedError()


class DataDispatcher():
    def __init__(self,
                 data_handler: DataHandler,
                 n: int=0, #number of clients
                 eval_on_user: bool=True):
        assert(data_handler.size() >= n)
        if n <= 1: n = data_handler.size()
        self.data_handler = data_handler
        self.n = n
        self.tr_assignments = [[] for _ in range(n)]
        self.te_assignments = [[] for _ in range(n)]
        for i in range(data_handler.size()):
            self.tr_assignments[i % n].append(i)
        self.eval_on_user = eval_on_user
        if eval_on_user:
            for i in range(data_handler.eval_size()):
                self.te_assignments[i % n].append(i)

    def __getitem__(self, idx: int) -> Any:
        assert(0 <= idx < self.n), "Index %d out of range." %idx
        return self.data_handler.at(self.tr_assignments[idx]), \
               self.data_handler.at(self.te_assignments[idx], True)
    
    def size(self) -> int:
        return self.n

    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.data_handler.get_eval_set()
    
    def has_test(self) -> bool:
        return self.data_handler.eval_size() > 0

#TODO: download and add new datasets
def load_classification_dataset(name: str,
                                path: str=None,
                                normalize: bool=True,
                                as_tensor: bool=True) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    if name == "iris":
        dataset = datasets.load_iris()
        X, y = dataset.data, dataset.target
    elif name == "breast":
        dataset = datasets.load_breast_cancer()
        X, y = dataset.data, dataset.target
    elif name in {"sonar", "ionosphere", "abalone", "banknote", "diabetes"}:
        X, y = load_svmlight_file(os.path.join(path, name + ".svmlight"))
        X = X.toarray()
    else:
        raise ValueError("Unknown dataset %s." %name)

    #if set(y) == 2:
    #    y = np.array([0 if yy <= 0 else 1 for yy in y])

    if normalize:
        X = StandardScaler().fit_transform(X)

    if as_tensor:
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()#.reshape(y.shape[0], 1)

    return X, y

#TODO: download
def load_recsys_dataset(name: str,
                        path: str) -> Dict[int, List[Tuple[int, float]]]:
    ratings = {}
    if name == "ml100k" or name == "ml1m":
        with open(os.path.join(path, name + ".txt"), "r") as f:
            for line in f.readlines():
                u, i, r = list(map(int, line.strip().split(",")))
                if u not in ratings:
                    ratings[u] = []
                ratings[u].append((i, r))
    else:
        raise ValueError("Unknown dataset %s." %name)
    return ratings