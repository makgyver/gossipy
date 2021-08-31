import os
from typing import Any, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import torch
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .. import LOG

__all__ = ["DataHandler", "DataDispatcher", "load_classification_dataset"]


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


UCI_URL_AND_CLASS = {
    "spambase" : ("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", 57),
    "sonar" : ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", 60),
    "ionosphere" : ("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data", 34),
    "abalone" : ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", 0),
    "banknote" : ("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", 4)
}


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
    #TODO: elif add more sklearn datasets
    #
    elif name in {"sonar", "ionosphere", "abalone", "banknote", "spambase"}:
        #if path is not None:
        #    X, y = load_svmlight_file(os.path.join(path, name + ".svmlight"))
        #    X = X.toarray()
        #else:
        url, label_id = UCI_URL_AND_CLASS[name]
        LOG.info("Downloading dataset %s from '%s'." %(name, url))
        data = pd.read_csv(url, header=None).to_numpy()
        y = LabelEncoder().fit_transform(data[:, label_id])
        X = np.delete(data, [label_id], axis=1).astype('float64')
    else:
        raise ValueError("Unknown dataset %s." %name)

    if normalize:
        X = StandardScaler().fit_transform(X)

    if as_tensor:
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()#.reshape(y.shape[0], 1)

    return X, y

#TODO: download
# def load_recsys_dataset(name: str,
#                         path: str) -> Dict[int, List[Tuple[int, float]]]:
#     ratings = {}
#     if name == "ml100k" or name == "ml1m":
#         with open(os.path.join(path, name + ".txt"), "r") as f:
#             for line in f.readlines():
#                 u, i, r = list(map(int, line.strip().split(",")))
#                 if u not in ratings:
#                     ratings[u] = []
#                 ratings[u].append((i, r))
#     else:
#         raise ValueError("Unknown dataset %s." %name)
#     return ratings