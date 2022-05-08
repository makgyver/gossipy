import numpy as np
import torch
from typing import Any, Tuple, Union, List, Dict, Optional
from sklearn.model_selection import train_test_split
from . import DataHandler

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = [
    "ClassificationDataHandler",
    "ClusteringDataHandler",
    "RegressionDataHandler",
    "RecSysDataHandler"
]


class ClassificationDataHandler(DataHandler):
    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 X_te: Optional[Union[np.ndarray, torch.Tensor]]=None,
                 y_te: Optional[Union[np.ndarray, torch.Tensor]]=None,
                 test_size: float=0.2,
                 seed: int=42):
        assert(0 <= test_size < 1)
        assert(isinstance(X, (torch.Tensor, np.ndarray)))

        if test_size > 0 and (X_te is None or y_te is None):
            if isinstance(X, torch.Tensor):
                n: int = X.shape[0]
                te: int = round(n * test_size)
                torch.manual_seed(seed)
                perm = torch.randperm(n)
                split = perm[:n-te], perm[n-te:]
                self.Xtr, self.ytr = X[split[0], :], y[split[0]]
                self.Xte, self.yte = X[split[1], :], y[split[1]]
            else:
                self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(X, y,
                                                                          test_size=test_size,
                                                                          random_state=seed,
                                                                          shuffle=True)
        else:
            self.Xtr, self.ytr = X, y
            self.Xte, self.yte = X_te, y_te

    # CHECKME: typing
    def __getitem__(self, idx: Union[int, List[int]])-> \
                    Union[Tuple[np.ndarray, Union[int, List[int]]], \
                          Tuple[torch.Tensor, Union[int, List[int]]]]:
        return self.Xtr[idx, :], self.ytr[idx]
    
    def at(self, 
           idx: Union[int, List[int]],
           eval_set=False) -> Tuple[np.ndarray, int]:
        if eval_set:
            if (not isinstance(idx, list) or idx):
                return self.Xte[idx, :], self.yte[idx]
            else: return None
        else: return self[idx]

    def size(self, dim: int=0) -> int:
        return self.Xtr.shape[dim]
    
    def get_train_set(self) -> Tuple[Any, Any]:
        return self.Xtr, self.ytr

    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.Xte, self.yte
    
    def eval_size(self) -> int:
        return self.Xte.shape[0] if self.Xte is not None else 0


# Same as ClassificationDataHandler but without a test set
class ClusteringDataHandler(ClassificationDataHandler):
    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 seed: int=42):
        super(ClusteringDataHandler, self).__init__(X, y, 0, seed)

    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.get_train_set()
    
    def eval_size(self) -> int:
        return self.size()

# Same as ClassificationDataHandler but with float labels
# Alternative: creating a unique DataHandler class for both classification and regression
class RegressionDataHandler(ClassificationDataHandler):
    def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[np.ndarray, float]:
        return self.Xtr[idx, :], self.ytr[idx]
    
    def at(self, 
           idx: Union[int, List[int]],
           eval_set=False) -> Tuple[np.ndarray, float]:
        super().at(idx, eval_set)


class RecSysDataHandler(DataHandler):
    def __init__(self,
                 ratings: Dict[int, List[Tuple[int, float]]],
                 n_users: int,
                 n_items: int,
                 test_size: float=0.2,
                 seed: int=42):
        self.ratings = ratings
        self.n_users = n_users
        self.n_items = n_items
        self.test_id = []
        np.random.seed(seed)
        for u in range(len(self.ratings)):
            self.test_id.append(max(1, int(len(self.ratings[u]) * (1 - test_size))))
            self.ratings[u] = np.random.permutation(self.ratings[u])

    def __getitem__(self, idx: int) -> List[Tuple[int, float]]:
        return self.ratings[idx][:self.test_id[idx]]

    def at(self, 
           idx: int,
           eval_set: bool=False) -> List[Tuple[int, float]]:
        if eval_set:
            return self.ratings[idx][self.test_id[idx]:]
        else: return self[idx]

    def size(self):
        return self.n_users
    
    def get_train_set(self) -> Tuple[Any, Any]:
        return {u : self[u] for u in range(self.n_users)}

    def get_eval_set(self) -> Tuple[Any, Any]:
        return {u : self.at(u, True) for u in range(self.n_users)}
    
    def eval_size(self) -> int:
        return 0