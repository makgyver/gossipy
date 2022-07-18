import numpy as np
import torch
from typing import Any, Tuple, Union, List, Dict, Optional
from sklearn.model_selection import train_test_split
from . import DataHandler

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache License, Version 2.0"
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
                 X_te: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 y_te: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 test_size: float = 0.2,
                 seed: int = 42,
                 on_device: bool = False):
        """Handler for classification data.

        The handlers provides methods to access the data and to split it into
        training and evaluation sets.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            The data set examples matrix. If ``X_te`` is not None, then the
            data set is assumed to be already splitted into training and evaluation set
            (``test_size`` will be ignored).
        y : np.ndarray or torch.Tensor
            The data set labels.
        X_te : np.ndarray or torch.Tensor, default=None
            The evaluation set examples matrix.
        y_te : np.ndarray or torch.Tensor, default=None
            The evaluation set labels.
        test_size : float, default=0.2
            The size of the evaluation set as a fraction of the data set.
        seed : int, default=42
            The seed used to split the data set into training and evaluation set.
        on_device : bool, default=False
            Whether data will be stored on the CUDA GPU if any
        """

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

        self.n_classes = len(np.unique(self.ytr))

        if on_device:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.Xtr = self.Xtr.to(self.device)
            self.ytr = self.ytr.to(self.device)
            self.Xte = self.Xte.to(self.device)
            self.yte = self.yte.to(self.device)
        else:
            device = "cpu"

    # CHECKME: typing

    def __getitem__(self, idx: Union[int, List[int]]) -> \
        Union[Tuple[np.ndarray, Union[int, List[int]]],
              Tuple[torch.Tensor, Union[int, List[int]]]]:
        return self.Xtr[idx, :], self.ytr[idx]

    def at(self,
           idx: Union[int, List[int]],
           eval_set=False) -> Tuple[np.ndarray, int]:
        """Get the data set example and label at the given index or list of indices.

        Parameters
        ----------
        idx : int or list of int
            The index or list of indices of the data set examples to get.
        eval_set : bool, default=False
            If True, the data set example and label are retrieved from the evaluation set.
            Otherwise, they are retrieved from the training set.

        Returns
        -------
        X : np.ndarray
            The data set example.
        y : int
            The data set label.
        """

        if eval_set:
            if (not isinstance(idx, list) or idx):
                return self.Xte[idx, :], self.yte[idx]
            else:
                return None
        else:
            return self[idx]

    # docstr-coverage:inherited
    def size(self, dim: int = 0) -> int:
        return self.Xtr.shape[dim]

    # docstr-coverage:inherited
    def get_train_set(self) -> Tuple[Any, Any]:
        return self.Xtr, self.ytr

    # docstr-coverage:inherited
    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.Xte, self.yte

    # docstr-coverage:inherited
    def eval_size(self) -> int:
        return self.Xte.shape[0] if self.Xte is not None else 0

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        res: str = f"{self.__class__.__name__}(size_tr={self.size()}, size_te={self.eval_size()}"
        res += f", n_feats={self.size(1)}, n_classes={self.n_classes})"
        return res


# Same as ClassificationDataHandler but without a test set
class ClusteringDataHandler(ClassificationDataHandler):
    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 on_device: bool = False):
        """Handler for clustering (unsupervised) data.

        The handlers provides methods to access the data. The evaluation set is the training set.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data set examples matrix.
        y : Union[np.ndarray, torch.Tensor]
            The data set labels.
        on_device : bool, default=False
            Whether data will be stored on the CUDA GPU if any
        """
        super(ClusteringDataHandler, self).__init__(
            X, y, 0, on_device=on_device)

    # docstr-coverage:inherited
    def get_eval_set(self) -> Tuple[Any, Any]:
        return self.get_train_set()

    # docstr-coverage:inherited
    def eval_size(self) -> int:
        return self.size()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size()})"

# Same as ClassificationDataHandler but with float labels
# Alternative: creating a unique DataHandler class for both classification and regression


class RegressionDataHandler(ClassificationDataHandler):
    """Same as :class:`ClassificationDataHandler` but with float labels."""

    def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[np.ndarray, float]:
        return self.Xtr[idx, :], self.ytr[idx]

    # docstr-coverage:inherited
    def at(self,
           idx: Union[int, List[int]],
           eval_set=False) -> Tuple[np.ndarray, float]:
        super().at(idx, eval_set)


class RecSysDataHandler(DataHandler):
    def __init__(self,
                 ratings: Dict[int, List[Tuple[int, float]]],
                 n_users: int,
                 n_items: int,
                 test_size: float = 0.2,
                 seed: int = 42):
        """Handler for recommendation system data.

        The handlers provides methods to access the rating data.

        Parameters
        ----------
        ratings : Dict[int, List[Tuple[int, float]]]
            The user-item ratings.
        n_users : int
            The number of users.
        n_items : int
            The number of items.
        test_size : float, default=0.2
            The size of the evaluation set as a fraction of the data set. The division is performed
            user-wise, i.e., for each user a subset of its ratings is selected as the evaluation set.
        seed : int, default=42
            The seed used to split the data set into training and evaluation set.
        """

        self.ratings = ratings
        self.n_users = n_users
        self.n_items = n_items
        self.test_id = []
        np.random.seed(seed)
        for u in range(len(self.ratings)):
            self.test_id.append(
                max(1, int(len(self.ratings[u]) * (1 - test_size))))
            self.ratings[u] = np.random.permutation(self.ratings[u])

    def __getitem__(self, idx: int) -> List[Tuple[int, float]]:
        return self.ratings[idx][:self.test_id[idx]]

    # docstr-coverage:inherited
    def at(self,
           idx: int,
           eval_set: bool = False) -> List[Tuple[int, float]]:
        if eval_set:
            return self.ratings[idx][self.test_id[idx]:]
        else:
            return self[idx]

    # docstr-coverage:inherited
    def size(self) -> int:
        return self.n_users

    # docstr-coverage:inherited
    def get_train_set(self) -> Tuple[Any, Any]:
        return {u: self[u] for u in range(self.n_users)}

    # docstr-coverage:inherited
    def get_eval_set(self) -> Tuple[Any, Any]:
        return {u: self.at(u, True) for u in range(self.n_users)}

    # docstr-coverage:inherited
    def eval_size(self) -> int:
        return 0

    def __str__(self) -> str:
        n_rat = sum([len(self.ratings[u]) for u in range(self.n_users)])
        return f"{self.__class__.__name__}(n_users={self.size()}, n_items={self.n_items}, n_ratings={n_rat}))"
