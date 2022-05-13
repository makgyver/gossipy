"""This module contains functions and classes to manage datasets loading and dispatching."""
import os
from typing import Any, Tuple, Union, Dict, List, Optional
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torchvision
from torch import Tensor, tensor
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .. import LOG
from ..utils import download_and_unzip, download_and_untar


# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["DataHandler",
           "DataDispatcher",
           "RecSysDataDispatcher",
           "load_classification_dataset",
           "load_recsys_dataset",
           "get_CIFAR10",
           "get_FashionMNIST",
           "get_FEMNIST"]

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

UCI_URL_AND_CLASS = {
    "spambase" : (UCI_BASE_URL + "spambase/spambase.data", 57),
    "sonar" : (UCI_BASE_URL + "undocumented/connectionist-bench/sonar/sonar.all-data", 60),
    "ionosphere" : (UCI_BASE_URL + "ionosphere/ionosphere.data", 34),
    "abalone" : (UCI_BASE_URL + "abalone/abalone.data", 0),
    "banknote" : (UCI_BASE_URL + "00267/data_banknote_authentication.txt", 4),
    #"dexter" : (UCI_BASE_URL + "dexter/DEXTER/", -1)
}


class DataHandler():
    def __init__(self):
        """Abstract class for data handlers.

        A :class:`DataHandler` class provides attributes and methods to manage a dataset.
        A subclass of :class:`DataHandler` must implement the following methods:

        - __getitem__(self, idx)
        - at(self, idx, eval_set=False)
        - size(self, dim: int=0)
        - get_eval_set(self)
        - get_train_set(self)
        - eval_size(self)
        """

        pass
        
    def __getitem__(self, idx: Union[int, List[int]]) -> Any:
        """Get a sample (or samples) from the training set.
        
        Parameters
        ----------
        idx : int or list[int]
            The index or indices of the sample(s) to get.
        
        Returns
        -------
        Any
            The sample(s) at the given index(ices) in the training set.
        """

        raise NotImplementedError()
    
    def at(self, 
           idx: Union[int, List[int]],
           eval_set: bool=False) -> Any:
        """Get a sample (or samples) from the training/test set.
        
        Parameters
        ----------
        idx : int or list[int]
            The index or indices of the sample(s) to get.
        eval_set : bool, default=False
            Whether to get the sample(s) from the training or the evaluation set.
        
        Returns
        -------
        Any
            The sample(s) at the given index(ices) in the training/evaluation set.
        """

        raise NotImplementedError()

    def size(self, dim: int=0) -> int:
        """Get the size of the training set along a given dimension.

        Parameters
        ----------
        dim : int, default=0
            The dimension along which to get the size of the dataset.

        Returns
        -------
        int
            The size of the dataset along the given dimension.
        """

        raise NotImplementedError()

    def get_eval_set(self) -> Tuple[Any, Any]:
        """Get the evaluation set of the dataset.

        Returns
        -------
        tuple[Any, Any]
            The evaluation set of the dataset.
        """

        raise NotImplementedError()
    
    def get_train_set(self) -> Tuple[Any, Any]:
        """Get the training set of the dataset.

        Returns
        -------
        tuple[Any, Any]
            The training set of the dataset.
        """

        raise NotImplementedError()

    def eval_size(self) -> int:
        """Get the number of examples of the evaluation set.

        Returns
        -------
        int
            The size of the evaluation set of the dataset.
        """

        raise NotImplementedError()


class DataDispatcher():
    def __init__(self,
                 data_handler: DataHandler,
                 n: int=0, #number of clients
                 eval_on_user: bool=True):
        """DataDispatcher is responsible for assigning data to clients.

        The assignment is done by shuffling the data and assigning it uniformly to the clients.
        If a specific assignment is required, use the `set_assignments` method.

        Parameters
        ----------
        data_handler : DataHandler
            The data handler that contains the data to be distributed.
        n : int, default=0
            The number of clients. If 0, the number of clients is set to the number of
            examples in the training set.
        eval_on_user : bool, default=True
            If True, a test set is assigned to each user.
        """

        assert(data_handler.size() >= n)
        if n <= 1: n = data_handler.size()
        self.data_handler = data_handler
        self.n = n
        self.eval_on_user = eval_on_user
        self.tr_assignments = None
        self.te_assignments = None
        #self.assign()
    
    def set_assignments(self, tr_assignments: List[int],
                              te_assignments: Optional[List[int]]) -> None:
        """Set the specified assignments for the training and test sets.
        
        The assignment must be provided as a list of integers with the same length as the
        number of examples in the training/test set. Each integer is the index of the client
        that will receive the example.

        Parameters
        ----------
        tr_assignments : list[int]
            The list of assignments for the training set.
        te_assignments : list[int], default=None
            The list of assignments for the test set. If None, the test set is not assigned.
        """

        assert len(tr_assignments) == self.n
        assert len(te_assignments) == self.n or not te_assignments
        self.tr_assignments = tr_assignments
        if te_assignments:
            self.te_assignments = te_assignments
        else:
            self.te_assignments = [[] for _ in range(self.n)]


    def assign(self, seed: int=42) -> None:
        """Assign the data to the clients.

        The assignment is done by shuffling the data and assigning it uniformly to the clients.

        Parameters
        ----------
        seed : int, default=42
            The seed for the random number generator.
        """

        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        torch.manual_seed(seed)
        iperm = torch.randperm(self.data_handler.size()).tolist()

        for i in range(self.data_handler.size()):
            self.tr_assignments[i % self.n].append(iperm[i])
        
        if self.eval_on_user:
            iperm = torch.randperm(self.data_handler.eval_size()).tolist()
            for i in range(self.data_handler.eval_size()):
                self.te_assignments[i % self.n].append(iperm[i])


    def __getitem__(self, idx: int) -> Any:
        """Return the data for the specified client.

        Parameters
        ----------
        idx : int
            The index of the client.
        
        Returns
        -------
        Any
            The data to assign to the specified client.
        
        Raises
        ------
        AssertionError
            If the index is out of range, i.e., no clients has the specified index.
        """

        assert 0 <= idx < self.n, "Index %d out of range." %idx
        return self.data_handler.at(self.tr_assignments[idx]), \
               self.data_handler.at(self.te_assignments[idx], True)
    
    def size(self) -> int:
        """Return the number of clients.

        Returns
        -------
        int
            The number of clients.
        """

        return self.n

    def get_eval_set(self) -> Tuple[Any, Any]:
        """Return the entire test set.

        Returns
        -------
        tuple[Any, Any]
            The test set.
        """

        return self.data_handler.get_eval_set()
    
    def has_test(self) -> bool:
        """Return True if there is a test set.

        Returns
        -------
        bool
            Whether there is a test set or not.
        """

        return self.data_handler.eval_size() > 0
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return "DataDispatcher(handler=%s, n=%d, eval_on_user=%s)" \
                %(self.data_handler, self.n, self.eval_on_user)


class RecSysDataDispatcher(DataDispatcher):
    from .handler import RecSysDataHandler
    def __init__(self,
                 data_handler: RecSysDataHandler):
        self.data_handler = data_handler
        self.n = self.data_handler.n_users
        self.eval_on_user = True
    
    def assign(self, seed=42):
        torch.manual_seed(seed)
        self.assignments = torch.randperm(self.data_handler.size()).tolist()


    def __getitem__(self, idx: int) -> Any:
        assert(0 <= idx < self.n), "Index %d out of range." %idx
        return self.data_handler.at(self.assignments[idx]), \
               self.data_handler.at(self.assignments[idx], True)
    
    def size(self) -> int:
        return self.n

    def get_eval_set(self) -> Tuple[Any, Any]:
        return None
    
    def has_test(self) -> bool:
        return False
    
    def __str__(self) -> str:
        return f"RecSysDataDispatcher(handler={self.data_handler}, eval_on_user={self.eval_on_user})"


def load_classification_dataset(name_or_path: str,
                                normalize: bool=True,
                                as_tensor: bool=True) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                               Tuple[np.ndarray, np.ndarray]]:
    """Load a classification dataset.

    Dataset can be load from *svmlight* file or can be one of the following:
    iris, breast, digits, wine, reuters, spambase, sonar, ionosphere, abalone, banknote.

    Parameters
    ----------
    name_or_path : str
        The name of the dataset or the path to the dataset.
    normalize : bool, default=True
        Whether to normalize (standard scaling) the data or not.
    as_tensor : bool, default=True
        Whether to return the data as a tensor or as a numpy array.
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or tuple[np.ndarray, np.ndarray]
        A tuple containing the data and the labels with the specified type.
    """

    if name_or_path == "iris":
        dataset = datasets.load_iris()
        X, y = dataset.data, dataset.target
    elif name_or_path == "breast":
        dataset = datasets.load_breast_cancer()
        X, y = dataset.data, dataset.target
    elif name_or_path == "digits":
        dataset = datasets.load_digits()
        X, y = dataset.data, dataset.target
    elif name_or_path == "wine":
        dataset = datasets.load_wine()
        X, y = dataset.data, dataset.target
    elif name_or_path == "reuters":
        url = "http://download.joachims.org/svm_light/examples/example1.tar.gz"
        folder = download_and_untar(url)[0]
        X_tr, y_tr = load_svmlight_file(folder + "/train.dat")
        X_te, y_te = load_svmlight_file(folder + "/test.dat")
        X_te = np.pad(X_te.toarray(), [(0, 0), (0, 17)], mode='constant', constant_values=0)
        X = np.vstack([X_tr.toarray(), X_te])
        y = np.concatenate([y_tr, y_te])
        y = LabelEncoder().fit_transform(y)
        shutil.rmtree(folder)
    elif name_or_path in {"sonar", "ionosphere", "abalone", "banknote", "spambase"}:
        url, label_id = UCI_URL_AND_CLASS[name_or_path]
        LOG.info("Downloading dataset %s from '%s'." %(name_or_path, url))
        data = pd.read_csv(url, header=None).to_numpy()
        y = LabelEncoder().fit_transform(data[:, label_id])
        X = np.delete(data, [label_id], axis=1).astype('float64')
    else:
        X, y = load_svmlight_file(name_or_path)
        X = X.toarray()

    if normalize:
        X = StandardScaler().fit_transform(X)

    if as_tensor:
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()#.reshape(y.shape[0], 1)

    return X, y


# TODO: add other recsys datasets
def load_recsys_dataset(name: str,
                        path: str=".") -> Tuple[Dict[int, List[Tuple[int, float]]], int, int]:
    """Load a recsys dataset.

    Currently, only the following datasets are supported: ml-100k, ml-1m, ml-10m and ml-20m.
    
    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str, default="."
        The path in which to download the dataset.
    
    Returns
    -------
    tuple[dict[int, list[tuple[int, float]]], int, int]
        A tuple contining the ratings, the number of users and the number of items.
        Ratings are represented as a dictionary mapping user ids to a list of tuples (item id, rating).
    """

    ratings = {}
    if name in {"ml-100k", "ml-1m", "ml-10m", "ml-20m"}:
        folder = download_and_unzip("https://files.grouplens.org/datasets/movielens/%s.zip" %name)[0]
        if name == "ml-100k":
            filename = "u.data"
            sep = "\t"
        elif name == "ml-20m":
            filename = "ratings.csv"
            sep = ","
        else:
            filename = "ratings.dat"
            sep = "::"

        ucnt = 0
        icnt = 0
        with open(os.path.join(path, folder, filename), "r") as f:
            umap = {}
            imap = {}
            for line in f.readlines():
                u, i, r = list(line.strip().split(sep))[0:3]
                u, i, r = int(u), int(i), float(r)
                if u not in umap:
                    umap[u] = ucnt
                    ratings[umap[u]] = []
                    ucnt += 1
                if i not in imap:
                    imap[i] = icnt
                    icnt += 1
                ratings[umap[u]].append((imap[i], r))

        shutil.rmtree(folder)
    else:
        raise ValueError("Unknown dataset %s." %name)
    return ratings, ucnt, icnt


def get_CIFAR10(path: str="./data",
                as_tensor: bool=True) -> Union[Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]],
                                               Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
    """Returns the CIFAR10 dataset.

    The method downloads the dataset if it is not already present in `path`.
    
    Parameters
    ----------
    path : str, default="./data"
        Path to save the dataset, by default "./data".
    as_tensor : bool, default=True
        If True, the dataset is returned as a tuple of pytorch tensors.
        Otherwise, the dataset is returned as a tuple of numpy arrays.
        By default, True.
    
    Returns
    -------
    tuple[tuple[np.ndarray, list], tuple[np.ndarray, list]] or tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
        Tuple of training and test sets of the form :math:`(X_train, y_train), (X_test, y_test)`.
    """

    download = not Path(os.path.join(path, "/cifar-10-batches-py")).is_dir()
    train_set = torchvision.datasets.CIFAR10(root=path,
                                             train=True,
                                             download=download)
    test_set = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            download=download)
    if as_tensor:
        train_set = tensor(train_set.data).float().permute(0,3,1,2) / 255.,\
                    tensor(train_set.targets)
        test_set = tensor(test_set.data).float().permute(0,3,1,2) / 255.,\
                   tensor(test_set.targets)
    else:
        train_set = train_set.data, train_set.targets
        test_set = test_set.data, test_set.targets

    return train_set, test_set


def get_FashionMNIST(path: str="./data",
                     as_tensor: bool=True) -> Union[Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]],
                                                          Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
    r"""Returns the FashionMNIST dataset.

    The method downloads the dataset if it is not already present in `path`.

    Parameters
    ----------
    path : str, default="./data"
        Path to save the dataset, by default "./data".
    as_tensor : bool, default=True
        If True, the dataset is returned as a tuple of pytorch tensors.
        Otherwise, the dataset is returned as a tuple of numpy arrays.
        By default, True.

    Returns
    -------
    Tuple[Tuple[np.ndarray, list], Tuple[np.ndarray, list]] or Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        Tuple of training and test sets of the form 
        :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`.
    """

    download = not Path(os.path.join(path, "/FashionMNIST/raw/")).is_dir()
    train_set = torchvision.datasets.FashionMNIST(root=path,
                                                  train=True,
                                                  download=download)
    test_set = torchvision.datasets.FashionMNIST(root=path,
                                                 train=False,
                                                 download=download)
    if as_tensor:
        train_set = train_set.data / 255., train_set.targets
        test_set = test_set.data / 255., test_set.targets
    else:
        train_set = train_set.data.numpy() / 255., train_set.targets.numpy()
        test_set = test_set.data.numpy() / 255., test_set.targets.numpy()

    return train_set, test_set


def get_FEMNIST(path: str="./data") -> Tuple[Tuple[torch.Tensor, torch.Tensor, List[int]], \
                                             Tuple[torch.Tensor, torch.Tensor, List[int]]]:
    url = 'https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz'
    te_name, tr_name = download_and_untar(url, path)
    Xtr, ytr, ids_tr = torch.load(os.path.join(path, tr_name))
    Xte, yte, ids_te = torch.load(os.path.join(path, te_name))
    tr_assignment = []
    te_assignment = []
    sum_tr = sum_te = 0
    for i in range(len(ids_tr)):
        ntr, nte = ids_tr[i], ids_te[i]
        tr_assignment.append(list(range(sum_tr, sum_tr + ntr)))
        te_assignment.append(list(range(sum_te, sum_te + nte)))
    return (Xtr, ytr, tr_assignment), (Xte, yte, te_assignment)