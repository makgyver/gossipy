from __future__ import annotations
import copy

from gossipy.model.nn import AdaLine, Pegasos
import torch
from torch import LongTensor
from torch.nn import ParameterList, Parameter
import numpy as np
from typing import Any, Callable, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from .. import LOG, Sizeable, CreateModelMode, EqualityMixin
from . import TorchModel
from .sampling import TorchModelPartition, TorchModelSampling

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["ModelHandler", "TorchModelHandler"]


class ModelHandler(Sizeable, EqualityMixin):
    def __init__(self,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE,
                 *args, **kwargs):
        self.model = None
        self.mode = create_model_mode
        self.n_updates = 0

    def init(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def _update(self, data: Any, *args, **kwargs) -> None:
        raise NotImplementedError()

    def _merge(self, other_model_handler: ModelHandler, *args, **kwargs) -> None:
        raise NotImplementedError()

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

    def evaluate(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def copy(self) -> Any:
        return copy.deepcopy(self)
    
    def get_size(self) -> int:
        return self.model.get_size() if self.model is not None else 0


class TorchModelHandler(ModelHandler):
    def __init__(self,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 l2_reg: float=0.01,
                 learning_rate: float=0.001,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE,
                 copy_model=True):
        super(TorchModelHandler, self).__init__(create_model_mode)
        self.model = copy.deepcopy(net) if copy_model else net
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=l2_reg)
        self.criterion = criterion

    def init(self) -> None:
        self.model.init_weights()

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        self.model.train()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1

    def _merge(self, other_model_handler: TorchModelHandler) -> None:
        dict_params1 = self.model.state_dict()
        dict_params2 = other_model_handler.model.state_dict()

        for key in dict_params1:
            dict_params2[key] = (dict_params1[key] + dict_params2[key]) / 2.

        self.model.load_state_dict(dict_params2)
        self.n_updates = max(self.n_updates, other_model_handler.n_updates)

    def evaluate(self,
                 data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, int]:
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
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }

        if scores.shape[1] == 2:
            auc_scores = scores[:, 1].detach().cpu().numpy().flatten()
            if len(set(y_true)) == 2:
                res["auc"] = roc_auc_score(y_true, auc_scores).astype(float)
            else:
                res["auc"] = 0.5
                LOG.warning("*** WARNING: # of classes != 2. AUC is set to 0.5. ***")
        return res


class AdaLineHandler(ModelHandler):
    def __init__(self,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE,
                 copy_model: bool=True):
        super(AdaLineHandler, self).__init__(create_model_mode)
        self.model = copy.deepcopy(net) if copy_model else net
        self.learning_rate = learning_rate
    
    def init(self) -> None:
        self.model.init_weights()
    
    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        self.n_updates += len(y)
        for i in range(len(y)):
            self.model.model += self.learning_rate * (y[i] - self.model(x[i])) * x[i]
    
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
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc":  roc_auc_score(y_true, auc_scores).astype(float)
        }

        return res


class PegasosHandler(AdaLineHandler):
    def __init__(self,
                 net: Pegasos,
                 lam: float,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE,
                 copy_model: bool=True):
        super(PegasosHandler, self).__init__(net, lam, create_model_mode, copy_model)
    
    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        for i in range(len(y)):
            self.n_updates += 1
            lr = 1. / (self.n_updates * self.learning_rate)
            y_pred = self.model(x[i])
            self.model.model *= (1. - lr * self.learning_rate)
            self.model.model += ((y_pred * y[i] - 1) < 0).float() * (lr * y[i] * x[i])


class SamplingTMHMixin(ModelHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, create_model_mode=CreateModelMode.MERGE_UPDATE, **kwargs)
    
    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 sample: Dict[int, Optional[Tuple[LongTensor, ...]]]) -> None:
        self._merge(recv_model, sample)
        self._update(data)
    
    def _merge(self, other_model_handler: SamplingTMHMixin,
                     sample: Dict[int, Optional[Tuple[LongTensor, ...]]]) -> None:
        TorchModelSampling.merge(sample, self.model, other_model_handler.model)
        #TODO: update the number of updates


class SamplingTMH(SamplingTMHMixin, TorchModelHandler):
    def __init__(self, *args, **kwargs):
        if "create_model_mode" in kwargs: kwargs.pop('create_model_mode')
        if len(args) > 5: args[5] = CreateModelMode.MERGE_UPDATE
        super().__init__(*args, **kwargs)


class PartitionedTMH(TorchModelHandler):
    def __init__(self,
                 net: TorchModel,
                 tm_partition: TorchModelPartition,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 l2_reg: float=0.01,
                 learning_rate: float=0.001,
                 copy_model=True):
        super(PartitionedTMH, self).__init__(net,
                                             optimizer,
                                             criterion,
                                             l2_reg,
                                             learning_rate,
                                             CreateModelMode.MERGE_UPDATE,
                                             copy_model)
        self.tm_partition = tm_partition
        self.n_updates = np.array([0 for _ in range(tm_partition.n_parts)], dtype=int)
    
    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 id_part: int) -> None:
        self._merge(recv_model, id_part)
        self._update(data)
    
    def _merge(self, other_model_handler: PartitionedTMH, id_part: int) -> None:
        w = (self.n_updates[id_part], other_model_handler.n_updates[id_part])
        self.tm_partition.merge(id_part, self.model, other_model_handler.model, weights=w)
        self.n_updates[id_part] = max(self.n_updates[id_part],
                                      other_model_handler.n_updates[id_part])
    
    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.n_updates += 1
        x, y = data
        self.model.train()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self._adjust_gradient()
        self.optimizer.step()
    
    def _adjust_gradient(self):
        plist = ParameterList(self.model.parameters())
        with torch.no_grad():
            for p, t_ids in self.tm_partition.partitions.items():
                for i, par in enumerate(plist):
                    if t_ids[i] is not None:
                        par.grad[t_ids[i]] /= self.n_updates[p]