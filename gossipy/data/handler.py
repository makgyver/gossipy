import numpy as np
import torch
from torch.utils.data import random_split
from typing import Any, Tuple, Union, List
from sklearn.model_selection import train_test_split
from . import DataHandler

__all__ = ["ClassificationDataHandler"]


class ClassificationDataHandler(DataHandler):
    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 test_size: float=0.2,
                 seed: int=42):
        assert(0 <= test_size < 1)
        assert(isinstance(X, (torch.Tensor, np.ndarray)))

        if test_size > 0:
            if isinstance(X, torch.Tensor):
                n: int = X.shape[0]
                te: int = round(n * test_size)
                split = random_split(torch.LongTensor(range(n)),
                                    [n - te, te],
                                    generator=torch.Generator().manual_seed(seed))
                self.Xtr = X[split[0], :]
                self.ytr = y[split[0], :]
                self.Xte = X[split[1], :]
                self.yte = y[split[1], :]
            else:
                self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(X, y,
                                                                        test_size=test_size,
                                                                        random_state=seed,
                                                                        shuffle=True)
        else:
            self.Xtr = X
            self.Ytr = y
            self.Xte = self.yte = None

    def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[np.ndarray, int]:
        return self.Xtr[idx, :], self.ytr[idx]
    
    def at(self, idx: Union[int, List[int]], eval_set=False):
        if eval_set:
            if (not isinstance(idx, list) or idx): # CHECK: why not isinstance(...)?
                return self.Xte[idx, :], self.yte[idx]
            else: return None
        else: return self[idx]

    def size(self) -> int:
        return self.Xtr.shape[0]
    
    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.Xte, self.yte
    
    def eval_size(self) -> int:
        return self.Xte.shape[0] if self.Xte is not None else 0