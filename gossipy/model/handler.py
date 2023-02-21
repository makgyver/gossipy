from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import torch
from torch import LongTensor
from torch.nn import ParameterList, Parameter
import numpy as np
from typing import Any, Callable, Tuple, Dict, Optional, Union, Iterable
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from scipy.optimize import linear_sum_assignment as hungarian

from .. import CACHE, LOG, CacheKey, Sizeable
from ..core import CreateModelMode
from . import TorchModel
from .sampling import TorchModelPartition, TorchModelSampling
from .nn import AdaLine

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
    "ModelHandler",
    "TorchModelHandler",
    "AdaLineHandler",
    "PegasosHandler",
    "SamplingTMH",
    "PartitionedTMH",
    "MFModelHandler",
    "KMeansHandler"
]

# Undocumented class
class ModelEqualityMixin(object):
    
    # docstr-coverage:excused `internal class to handle equality between models`
    def __init__(self):
        pass

    # docstr-coverage:excused `internal class to handle equality between models`
    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    # docstr-coverage:excused `internal class to handle equality between models`
    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)



class ModelHandler(Sizeable, ModelEqualityMixin, ABC):
    def __init__(self,
                 create_model_mode: CreateModelMode=CreateModelMode.MERGE_UPDATE,
                 *args, **kwargs):
        """The ModelHandler class is the base class for all model handlers.

        A ModelHandler class manages the training (i.e., update) and evaluation of a model.
        It is also responsible for the merging of two (or more) models according to the
        :attr:`mode`.

        A ModelHandler is a callable object. Calling a ModelHandler instance will preform the
        update according to the :attr:`mode`.

        Parameters
        ----------
        create_model_mode : CreateModelMode, default=CreateModelMode.MERGE_UPDATE
            The mode in which the model is created/updated.
        
        See Also
        --------
        gossipy.core.CreateModelMode
        """

        self.model = None
        self.mode = create_model_mode
        self.n_updates = 0

    @abstractmethod
    def init(self, *args, **kwargs) -> None:
        """Initialize the model."""

        pass
    
    @abstractmethod
    def _update(self, data: Any, *args, **kwargs) -> None:
        """Update the model.
        
        The update usually consists of a number of training steps/epochs using the provided data.

        Parameters
        ----------
        data : Any
            The data to use for the update.
        """

        pass
    
    @abstractmethod
    def _merge(self, other_model_handler: ModelHandler, *args, **kwargs) -> None:
        """Merge the model handler with the provided model handler.

        Parameters
        ----------
        other_model_handler : ModelHandler
            The model handler to merge with.
        """

        pass

    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 *args,
                 **kwargs) -> None:
        if self.mode == CreateModelMode.UPDATE:
            recv_model._update(data)
            self.model = copy.deepcopy(recv_model.model)
            self.n_updates = recv_model.n_updates
        elif self.mode == CreateModelMode.MERGE_UPDATE:
            self._merge(recv_model)
            self._update(data)
        elif self.mode == CreateModelMode.UPDATE_MERGE:
            self._update(data)
            recv_model._update(data)
            self._merge(recv_model)
        elif self.mode == CreateModelMode.PASS:
            self.model = copy.deepcopy(recv_model.model)
        else:
            raise ValueError("Unknown create model mode %s" %str(self.mode))

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """Evaluate the model."""

        pass

    def copy(self) -> Any:
        """Return a deep copy of the model handler."""

        return copy.deepcopy(self)
    
    def get_size(self) -> int:
        """Return the size of the model.

        Returns
        -------
        int
            The size of the model.
        """

        return self.model.get_size() if self.model is not None else 0
    
    def caching(self, owner: int) -> CacheKey:
        """Cache the model handler and return the cache key.

        Parameters
        ----------
        owner : int
            The ID of the client that own of this particular model handler.
        
        Returns
        -------
        CacheKey
            The cache key corresponding to this model handler in the cache.
        """

        key = CacheKey(owner, self.n_updates)
        CACHE.push(key, self.copy())
        return key
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={str(self.model)}_{self.n_updates}, mode={self.mode})"
     

class TorchModelHandler(ModelHandler):
    def __init__(self,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int=1,
                 batch_size: int=32,
                 create_model_mode: CreateModelMode=CreateModelMode.MERGE_UPDATE,
                 copy_model=True):
        """Handler for torch models.

        This handler is responsible for the training and evaluation of a pytorch model. Thus it
        requires a :class:`~gossipy.core.torch.TorchModel` instance that represents the model to
        be trained, and an optimizer (e.g., :class:`torch.optim.SGD`) with its parameters (a dict).
        The ``criterion`` is the loss function to be used for the training.

        Parameters
        ----------
        net : TorchModel
            The model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for the training.
        optimizer_params : Dict[str, Any]
            The parameters of the optimizer.
        criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to be used for the training.
        local_epochs : int, default=1
            The number of local epochs.
        batch_size : int, default=32
            The batch size.
        create_model_mode : CreateModelMode, default=CreateModelMode.MERGE_UPDATE
            The mode in which the model is created/updated
        copy_model : bool, default=True
            Whether to use a copy of the model (i.e., ``net``) or not.
        """

        super(TorchModelHandler, self).__init__(create_model_mode)
        self.model = copy.deepcopy(net) if copy_model else net
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.criterion = criterion
        assert (batch_size == 0 and local_epochs > 0) or (batch_size > 0)
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def init(self) -> None:
        self.model.init_weights()

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        batch_size = x.size(0) if not self.batch_size else self.batch_size
        if self.local_epochs > 0:
            for _ in range(self.local_epochs):
                perm = torch.randperm(x.size(0))
                x, y = x[perm], y[perm]
                for i in range(0, x.size(0), batch_size):
                    self._local_step(x[i : i + batch_size], y[i : i + batch_size])
        else:
            perm = torch.randperm(x.size(0))
            self._local_step(x[perm][:batch_size], y[perm][:batch_size])
        self.n_updates += 1
    
    def _local_step(self, x:torch.Tensor, y:torch.Tensor) -> None:
        self.model.train()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _merge(self, other_model_handler: Union[TorchModelHandler, Iterable[TorchModelHandler]]) -> None:
        dict_params1 = self.model.state_dict()

        if isinstance(other_model_handler, TorchModelHandler):
            dicts_params2 = [other_model_handler.model.state_dict()]
            n_up = other_model_handler.n_updates
        else:
            dicts_params2 = [omh.model.state_dict() for omh in other_model_handler]
            n_up = max([omh.n_updates for omh in other_model_handler])

        # Perform the average overall models including its weights
        # CHECK: whether to allow the merging of the other models before the averaging 
        div = len(dicts_params2) + 1
        for key in dict_params1:
            for dict_params2 in dicts_params2:
                dict_params1[key] += dict_params2[key]
            dict_params1[key] /= div

        self.model.load_state_dict(dict_params1)
        # Gets the maximum number of updates from the merged models
        self.n_updates = max(self.n_updates, n_up) 

    def evaluate(self,
                 data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, int]:
        """Evaluates the model on the given data.

        Parameters
        ----------
        data : Tuple[torch.Tensor, torch.Tensor]
            The data to use for the evaluation.

        Returns
        -------
        Dict[str, int]
            The evaluation results. The dictionary keys are the metrics names, and the values are
            the corresponding scores.
        
        Notes
        -----
        Currently, only metrics for classification tasks are implemented. Specifically, 
        the evaluation metrics are: ``accuracy``, ``precision``, ``recall``, ``f1``, and,
        when possible, ``roc_auc``.
        """

        x, y = data
        self.model.eval()
        scores = self.model(x)

        if y.dim() == 1:
            y_true = y.cpu().numpy().flatten()
        else:
            y_true = torch.argmax(y, dim=-1).cpu().numpy().flatten()

        pred = torch.argmax(scores, dim=-1)
        y_pred = pred.cpu().numpy().flatten()
        
        res = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
            "recall": recall_score(y_true, y_pred, zero_division=0, average="macro"),
            "f1_score": f1_score(y_true, y_pred, zero_division=0, average="macro")
        }

        if scores.shape[1] == 2:
            auc_scores = scores[:, 1].detach().cpu().numpy().flatten()
            if len(set(y_true)) == 2:
                res["auc"] = roc_auc_score(y_true, auc_scores).astype(float)
            else:
                res["auc"] = 0.5
                LOG.warning("# of classes != 2. AUC is set to 0.5.")
        return res


class AdaLineHandler(ModelHandler):
    def __init__(self,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE,
                 copy_model: bool=True):
        """This class implements the AdaLine model handler.

        Parameters
        ----------
        net : AdaLine
            The AdaLine model.
        learning_rate : float
            The learning rate.
        create_model_mode : CreateModelMode, default=CreateModelMode.UPDATE
            The mode in which the model is created/updated.
        copy_model : bool, default=True
            Whether to use a copy of the model (i.e., ``net``) or not.
        """

        super(AdaLineHandler, self).__init__(create_model_mode)
        self.model = copy.deepcopy(net) if copy_model else net
        self.learning_rate = learning_rate
    
    def init(self) -> None:
        self.model.init_weights()
    
    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        self.n_updates += len(y)
        for i in range(len(y)):
            self.model.model += self.learning_rate * (y[i] - self.model(x[i:i+1])) * x[i]
    
    def _merge(self, other_model_handler: PegasosHandler) -> None:
        self.model.model = Parameter(0.5 * (self.model.model + other_model_handler.model.model),
                                     requires_grad=False)
        self.n_updates = max(self.n_updates, other_model_handler.n_updates)

    def evaluate(self,
                 data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, int]:
        x, y = data
        scores = self.model(x)
        y_true = y.cpu().numpy().flatten()
        y_pred = 2 * (scores >= 0).float().cpu().numpy().flatten() - 1
        auc_scores = scores.detach().cpu().numpy().flatten()

        res = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
            "recall": recall_score(y_true, y_pred, zero_division=0, average="macro"),
            "f1_score": f1_score(y_true, y_pred, zero_division=0, average="macro"),
            "auc":  roc_auc_score(y_true, auc_scores).astype(float)
        }

        return res


class PegasosHandler(AdaLineHandler):
    def __init__(self,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE,
                 copy_model: bool=True):
        """This class implements the Pegasos model handler.

        Parameters
        ----------
        net : AdaLine
            The Pegasos (same as AdaLine) model.
        learning_rate : float
            The learning rate.
        create_model_mode : CreateModelMode, default=CreateModelMode.UPDATE
            The mode in which the model is created/updated.
        copy_model : bool, default=True
            Whether to use a copy of the model (i.e., ``net``) or not.
        """

        super(PegasosHandler, self).__init__(net, learning_rate, create_model_mode, copy_model)
    
    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        for i in range(len(y)):
            self.n_updates += 1
            lr = 1. / (self.n_updates * self.learning_rate)
            y_pred = self.model(x[i:i+1])
            self.model.model *= (1. - lr * self.learning_rate)
            self.model.model += ((y_pred * y[i] - 1) < 0).float() * (lr * y[i] * x[i])


class SamplingTMH(TorchModelHandler):
    def __init__(self, sample_size: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
    
    def _merge(self, other_model_handler: SamplingTMH,
                     sample: Dict[int, Optional[Tuple[LongTensor, ...]]]) -> None:
        TorchModelSampling.merge(sample, self.model, other_model_handler.model)
    
    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 sample: Dict[int, Optional[Tuple[LongTensor, ...]]]) -> None:
        if self.mode == CreateModelMode.UPDATE:
            recv_model._update(data)
            self._merge(recv_model, sample)
        elif self.mode == CreateModelMode.MERGE_UPDATE:
            self._merge(recv_model, sample)
            self._update(data)
        elif self.mode == CreateModelMode.UPDATE_MERGE:
            self._update(data)
            recv_model._update(data)
            self._merge(recv_model, sample)
        elif self.mode == CreateModelMode.PASS:
            raise ValueError("Mode PASS not allowed for sampled models.")
        else:
            raise ValueError("Unknown create model mode %s." %str(self.mode))
        

class PartitionedTMH(TorchModelHandler):
    def __init__(self,
                 net: TorchModel,
                 tm_partition: TorchModelPartition,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int=1,
                 batch_size: int=32,
                 create_model_mode: CreateModelMode=CreateModelMode.MERGE_UPDATE,
                 copy_model=True):
        super(PartitionedTMH, self).__init__(net,
                                             optimizer,
                                             optimizer_params,
                                             criterion,
                                             local_epochs,
                                             batch_size,
                                             create_model_mode,
                                             copy_model)
        self.tm_partition = tm_partition
        self.n_updates = np.array([0 for _ in range(tm_partition.n_parts)], dtype=int)
    
    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 id_part: int) -> None:
        if self.mode == CreateModelMode.UPDATE:
            recv_model._update(data)
            self._merge(recv_model, id_part)
        elif self.mode == CreateModelMode.MERGE_UPDATE:
            self._merge(recv_model, id_part)
            self._update(data)
        elif self.mode == CreateModelMode.UPDATE_MERGE:
            self._update(data)
            recv_model._update(data)
            self._merge(recv_model, id_part)
        elif self.mode == CreateModelMode.PASS:
            raise ValueError("Mode PASS not allowed for partitioned models.")
        else:
            raise ValueError("Unknown create model mode %s." %str(self.mode))

    
    def _merge(self, other_model_handler: PartitionedTMH, id_part: int) -> None:
        w = (self.n_updates[id_part], other_model_handler.n_updates[id_part])
        self.tm_partition.merge(id_part, self.model, other_model_handler.model, weights=w)
        self.n_updates[id_part] = max(self.n_updates[id_part],
                                      other_model_handler.n_updates[id_part])
    
    def _local_step(self, x:torch.Tensor, y:torch.Tensor) -> None:
        self.model.train()
        self.n_updates += 1
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self._adjust_gradient()
        self.optimizer.step()
        
    def _adjust_gradient(self) -> None:
        plist = ParameterList(self.model.parameters())
        with torch.no_grad():
            for p, t_ids in self.tm_partition.partitions.items():
                for i, par in enumerate(plist):
                    if t_ids[i] is not None:
                        par.grad[t_ids[i]] /= self.n_updates[p]

    def caching(self, owner: int) -> CacheKey:
        key = CacheKey(owner, str(self.n_updates))
        CACHE.push(key, self.copy())
        return key


class MFModelHandler(ModelHandler):
    def __init__(self,
                 dim: int,
                 n_items: int,
                 lam_reg: float=0.1,
                 learning_rate: float=0.001,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE):
        super(MFModelHandler, self).__init__(create_model_mode)
        self.reg = lam_reg
        self.k = dim
        self.lr = learning_rate
        self.n_items = n_items
        self.n_updates = 1

    def init(self, r_min: int=1, r_max: int=5) -> None:
        mul = np.sqrt((r_max - r_min) / self.k)
        X = np.random.rand(1, self.k) * mul
        Y = np.random.rand(self.n_items, self.k) * mul
        b = r_min / 2.0
        c = np.ones(self.n_items) * r_min / 2.0
        self.model = ((X, b), (Y, c))

    def _update(self, data: torch.Tensor) -> None:
        (X, b), (Y, c) = self.model
        for i, r in data:
            i = int(i)
            err = (r - np.dot(X, Y[i].T) - b - c[i])[0]
            Y[i] = (1. - self.reg * self.lr) * Y[i] + self.lr * err * X
            X = (1. - self.reg * self.lr) * X + self.lr * err * Y[i]
            b += self.lr * err
            c[i] += self.lr * err
        self.model = ((X, b), (Y, c))
        self.n_updates += 1

    def _merge(self, other_model_handler: MFModelHandler) -> None:
        _, (Y1, c1) = other_model_handler.model
        (X, b), (Y, c) = self.model
        den = self.n_updates + other_model_handler.n_updates
        Y = (Y * self.n_updates + Y1 * other_model_handler.n_updates) / (2.0 * den)
        c = (c * self.n_updates + c1 * other_model_handler.n_updates) / (2.0 * den)
        self.model = (X, b), (Y, c)

    def evaluate(self, ratings) -> Dict[str, float]:
        (X, b), (Y, c) = self.model
        R = (np.dot(X, Y.T) + b + c)[0]
        return {"rmse" : np.sqrt(np.mean([(r - R[int(i)])**2 for i, r in ratings]))}
    
    def get_size(self) -> int:
        return self.k * (self.n_items + 1)


class KMeansHandler(ModelHandler):
    def __init__(self,
                 k: int,
                 dim: int,
                 alpha: float=0.1,
                 matching: str="naive", #"hungarian"
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE):
        assert matching in {"naive", "hungarian"}, "Invalid matching method."
        super(KMeansHandler, self).__init__(create_model_mode)
        self.k = k
        self.dim = dim
        self.matching = matching
        self.alpha = alpha
        #self._init_count = 0
    
    def init(self) -> None:
        self.model = torch.rand(size=(self.k, self.dim))
    
    # def _has_empty(self) -> bool:
    #     return self._init_count < self.k
    
    # def _add_centroid(self, x: torch.FloatTensor):
    #     self.model[self._init_count] += x.flatten()
    #     self._init_count += 1
    
    def _perform_clust(self, x: torch.FloatTensor) -> int:
        dists = torch.cdist(x, self.model, p=2)
        return torch.argmin(dists, dim=1)

    def _update(self, data: torch.FloatTensor) -> None:
        x, _ = data
        # if self._has_empty():
        #     self._add_centroid(x)
        # else:
        idx = self._perform_clust(x)
        self.model[idx] = self.model[idx] * (1 - self.alpha) + self.alpha * x
        self.n_updates += 1

    def _merge(self, other_model_handler: KMeansHandler) -> None:
        # if self._has_empty():
        #     i = 0
        #     while self._has_empty() and i < other_model_handler._init_count:
        #         self._add_centroid(other_model_handler.model[i])
        #         i += 1
        # elif not other_model_handler._has_empty():
        if self.matching == "naive":
            self.model = (self.model + other_model_handler.model) / 2
        elif self.matching == "hungarian":
            cm_torch = torch.cdist(self.model, other_model_handler.model)
            cost_matrix = cm_torch.cpu().detach().numpy()
            matching_idx = hungarian(cost_matrix)[0]
            self.model = (self.model + other_model_handler.model[matching_idx]) / 2
    
    def evaluate(self, data: Tuple[torch.FloatTensor, torch.LongTensor]) -> Dict[str, float]:
        X, y = data
        y_pred = self._perform_clust(X).cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        return {"nmi": nmi(y_true, y_pred)}
    
    def get_size(self) -> int:
        return self.k * self.dim


class WeightedTMH(TorchModelHandler):

    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 weights: Iterable[float]) -> None:
        if self.mode == CreateModelMode.UPDATE:
            recv_model._update(data)
            self.model = copy.deepcopy(recv_model.model)
            self.n_updates = recv_model.n_updates
        elif self.mode == CreateModelMode.MERGE_UPDATE:
            self._merge(recv_model, weights)
            self._update(data)
        elif self.mode == CreateModelMode.UPDATE_MERGE:
            self._update(data)
            if isinstance(recv_model, Iterable):
                for rm in recv_model:
                    rm._update(data)
            else:
                recv_model._update(data)
            self._merge(recv_model, weights)
        else:
            raise ValueError("Invalid create model mode %s for WeightedTMH." %str(self.mode))
            
    def _merge(self, 
               other_model_handler: Union[TorchModelHandler, Iterable[TorchModelHandler]],
               weights: Iterable[float]) -> None:
        
        dict_params1 = self.model.state_dict()

        if isinstance(other_model_handler, TorchModelHandler):
            dicts_params2 = [other_model_handler.model.state_dict()]
            n_up = other_model_handler.n_updates
        else:
            dicts_params2 = [omh.model.state_dict() for omh in other_model_handler]
            n_up = max([omh.n_updates for omh in other_model_handler])

        # Perform the average overall models including its weights
        # CHECK: whether to allow the merging of the other models before the averaging 
        for key in dict_params1:
            dict_params1[key] *= weights[0]
            for i, dict_params2 in enumerate(dicts_params2):
                dict_params1[key] += dict_params2[key] * weights[i + 1]

        self.model.load_state_dict(dict_params1)
        # Gets the maximum number of updates from the merged models
        self.n_updates = max(self.n_updates, n_up)

