from __future__ import annotations
import copy
import torch
import numpy as np
from typing import Any, Callable, Tuple, Dict
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from .. import Sizeable, CreateModelMode

__all__ = ["ModelHandler", "TorchModelHandler"]


class ModelHandler(Sizeable):
    def __init__(self,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE):
        self.model = None
        self.mode = create_model_mode

    def init(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def _update(self, data: Any) -> None:
        raise NotImplementedError()

    def _merge(self, other_model_handler: ModelHandler) -> None:
        raise NotImplementedError()

    def __call__(self,
                 recv_model: Any,
                 data: Any,
                 *args,
                 **kwargs) -> None:
        if self.mode == CreateModelMode.UPDATE:
            recv_model._update(data)
            self.model = copy.deepcopy(recv_model.model)
        elif self.mode == CreateModelMode.MERGE_UPDATE:
            self._merge(recv_model)
            self._update(data)
        elif self.mode == CreateModelMode.UPDATE_MERGE:
            self._update(data)
            recv_model._update(data)
            self._merge(recv_model)
        else:
            raise ValueError("Unknown create model mode %s" %str(self.mode))

    def evaluate(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def copy(self) -> Any:
        return copy.deepcopy(self)
    
    def get_size(self) -> int:
        return self.model.get_size()


class TorchModelHandler(ModelHandler):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 l2_reg: float=0.01,
                 learning_rate: float=0.001,
                 create_model_mode: CreateModelMode=CreateModelMode.UPDATE):
        super(TorchModelHandler, self).__init__(create_model_mode)
        self.model = copy.deepcopy(net)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=l2_reg)
        self.criterion = criterion

    def init(self) -> None:
        self.model.init_weights()

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _merge(self, other_model_handler: TorchModelHandler) -> None:
        dict_params1 = self.model.state_dict()
        dict_params2 = other_model_handler.model.state_dict()

        for key in dict_params1:
            dict_params2[key] = (dict_params1[key] + dict_params2[key]) / 2.

        self.model.load_state_dict(dict_params2)

    def evaluate(self,
                 data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, int]:
        x, y = data
        scores = self.model(x)

        y_true = y.cpu().numpy().flatten()
        pred = torch.ones_like(scores)
        pred[scores <= .5] = 0
        y_pred = pred.cpu().numpy().flatten()
        auc_scores = scores.detach().cpu().numpy().flatten()

        res = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, auc_scores).astype(float) if len(set(y_true)) > 1 else .5,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
        return res
