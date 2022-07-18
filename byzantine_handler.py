from __future__ import annotations
from typing import Tuple
import torch
from gossipy.core import CreateModelMode
from gossipy.model.handler import ModelHandler, TorchModelHandler, PegasosHandler
import copy

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = [
    "AvoidReportMixin",
    "RandomGradientAttackMixin",
    "SameValueAttackMixin",
    "GradientScalingAttackMixin",
    "BackGradientAttackMixin",
    "RandomModelAttackMixin",
    "RandomFullModelAttackMixin",
    "TorchModelRandomGradientAttackHandler",
    "TorchModelSameValueAttackHandler",
    "TorchModelGradientScalingAttackHandler",
    "TorchModelBackGradientAttackHandler",
    "TorchModelRandomModelAttackHandler",
    "TorchModelRandomFullModelAttackHandler"
    "PegasosRandomGradientAttackHandler",
    "PegasosSameValueAttackHandler",
    "PegasosGradientScalingAttackHandler",
    "PegasosBackGradientAttackHandler",
    "PegasosRandomModelAttackHandler",
    "PegasosRandomFullModelAttackHandler"
]


class AvoidReportMixin():
    '''Inheriting objects won't be taken into account in the evaluation when using compatible SimulationReport (e.g. ByzantineSimulationReport).
    Usefull to avoir malicious clients voluntarily impacting results.

    WARNING : If there are too much client evaluating to None and evaluation sampling is too low, global evaluation may lead to a None value...
    '''

    def evaluate(self, *args, **kwargs):
        return None


class RandomGradientAttackMixin(AvoidReportMixin, TorchModelHandler):
    def __init__(self, scale: float):
        '''Adds a random vector to the model at each update.
        Acts like a getting a random gradient at each update.

        Parameters
        ----------
        scale: float
            The amplitude of noise
        '''
        self.scale = scale

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn(
                    param.size(), device=self.device) * self.scale)
        self.n_updates += 1


class RandomModelAttackMixin(AvoidReportMixin, TorchModelHandler):
    '''Each update replaces the model with a random model following a normal law with same mean and standard deviation then the previous one.
    '''

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        with torch.no_grad():
            state = self.model.state_dict()
            for name in state:
                std, mean = torch.std_mean(state[name])
                state[name] = torch.rand(
                    state[name].size(), device=self.device) * std + mean
        self.model.load_state_dict(state)
        self.n_updates += 1


class RandomFullModelAttackMixin(AvoidReportMixin, TorchModelHandler):
    '''Each update replaces the model with a random model following an uniform law with a given mean and scale
    Range of weights and biases are [mean - scale / 2, mean + scale / 2 ]

    Parameters
    ----------
    scale: float
        The amplitude of noise
    mean: float
        Mean of weights and biases
    '''

    def __init__(self, scale: float = 1.0, mean: float = 0.0):
        self.scale = scale
        self.mean = mean

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        with torch.no_grad():
            state = self.model.state_dict()
            for name in state:
                state[name] = torch.rand(
                    state[name].size(), device=self.device) * self.scale + (self.mean - self.scale / 2.)
        self.model.load_state_dict(state)
        self.n_updates += 1


class SameValueAttackMixin(AvoidReportMixin, ModelHandler):
    '''Does nothing at each update.
    Acts like a getting a null gradient at each update.
    '''

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        pass


class GradientScalingAttackMixin(AvoidReportMixin, TorchModelHandler):
    def __init__(self, scale: float):
        '''Scales the gradient of each update by a given factor

        Parameters
        ----------
        scale: float
            The scale used.
        '''
        self.scale = scale

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        self.model.train()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad *= self.scale
        self.optimizer.step()
        self.n_updates += 1


class BackGradientAttackMixin(GradientScalingAttackMixin):
    def __init__(self):
        '''Inverts the gradient of each update.
        '''
        GradientScalingAttackMixin.__init__(self, scale=-1.)


class TorchModelBackGradientAttackHandler(BackGradientAttackMixin, TorchModelHandler):
    def __init__(self,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True,
                 on_device=False):
        BackGradientAttackMixin.__init__(self)
        TorchModelHandler.__init__(self, net, optimizer, optimizer_params, criterion,
                                   local_epochs, batch_size, create_model_mode, copy_model, on_device)


class TorchModelRandomGradientAttackHandler(RandomGradientAttackMixin, TorchModelHandler):
    def __init__(self,
                 scale: float,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True,
                 on_device=False):
        RandomGradientAttackMixin.__init__(self, scale)
        TorchModelHandler.__init__(self, net, optimizer, optimizer_params, criterion,
                                   local_epochs, batch_size, create_model_mode, copy_model, on_device)


class TorchModelGradientScalingAttackHandler(GradientScalingAttackMixin, TorchModelHandler):
    def __init__(self,
                 scale: float,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True,
                 on_device=False):
        GradientScalingAttackMixin.__init__(self, scale)
        TorchModelHandler.__init__(self, net, optimizer, optimizer_params, criterion,
                                   local_epochs, batch_size, create_model_mode, copy_model, on_device)


class TorchModelRandomModelAttackHandler(RandomModelAttackMixin, TorchModelHandler):
    def __init__(self,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True,
                 on_device=False):
        TorchModelHandler.__init__(self, net, optimizer, optimizer_params, criterion,
                                   local_epochs, batch_size, create_model_mode, copy_model, on_device)


class TorchModelRandomFullModelAttackHandler(RandomFullModelAttackMixin, TorchModelHandler):
    def __init__(self,
                 scale: float,
                 mean: float,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True,
                 on_device=False):
        RandomFullModelAttackMixin.__init__(self, scale, mean)
        TorchModelHandler.__init__(self, net, optimizer, optimizer_params, criterion,
                                   local_epochs, batch_size, create_model_mode, copy_model, on_device)


class TorchModelSameValueAttackHandler(SameValueAttackMixin, TorchModelHandler):
    def __init__(self,
                 net: TorchModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_params: Dict[str, Any],
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 create_model_mode: CreateModelMode = CreateModelMode.MERGE_UPDATE,
                 copy_model=True,
                 on_device=False):
        TorchModelHandler.__init__(self, net, optimizer, optimizer_params, criterion,
                                   local_epochs, batch_size, create_model_mode, copy_model, on_device)


class PegasosSameValueAttackHandler(SameValueAttackMixin, PegasosHandler):
    def __init__(self,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True,
                 on_device: bool = False):
        PegasosHandler.__init__(self, net, learning_rate,
                                create_model_mode, copy_model, on_device)


class PegasosGradientScalingAttackHandler(AvoidReportMixin, PegasosHandler):
    def __init__(self,
                 scale: float,
                 nb: int,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True,
                 on_device: bool = False):
        PegasosHandler.__init__(self,
                                net, learning_rate, create_model_mode, copy_model, on_device)
        self.scale = scale
        self.nb = nb

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        for i in range(len(y)):
            self.n_updates += self.nb
            lr = 1. / (self.n_updates * self.learning_rate)
            y_pred = self.model(x[i:i+1])
            # self.model.model *= (1. - lr * self.learning_rate)
            # self.model.model += ((y_pred * y[i] - 1)
            #                     < 0).float() * (lr * y[i] * x[i])
            self.model.model += self.scale * (((y_pred * y[i] - 1) < 0) * (
                lr * y[i] * x[i]) - lr * self.learning_rate * self.model.model)


class PegasosRandomGradientAttackHandler(AvoidReportMixin, PegasosHandler):
    def __init__(self,
                 scale: float,
                 nb: float,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True,
                 on_device: bool = False):
        PegasosHandler.__init__(self,
                                net, learning_rate, create_model_mode, copy_model, on_device)
        self.scale = scale
        self.nb = nb

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        for i in range(self.nb):
            self.n_updates += self.nb
            self.model.model += self.scale * \
                torch.randn(self.model.model.size(), device=self.device)


class PegasosRandomModelAttackHandler(AvoidReportMixin, PegasosHandler):
    def __init__(self,
                 nb: int,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True,
                 on_device: bool = False):
        PegasosHandler.__init__(self,
                                net, learning_rate, create_model_mode, copy_model, on_device)
        self.nb = nb

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.n_updates += self.nb
        with torch.no_grad():
            state = self.model.state_dict()
            for name in state:
                std, mean = torch.std_mean(state[name])
                state[name] = torch.rand(
                    state[name].size(), device=self.device) * std + mean
        self.model.load_state_dict(state)


class PegasosRandomFullModelAttackHandler(AvoidReportMixin, PegasosHandler):
    def __init__(self,
                 scale: float,
                 mean: float,
                 nb: int,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True,
                 on_device: bool = False):
        PegasosHandler.__init__(self,
                                net, learning_rate, create_model_mode, copy_model, on_device)
        self.nb = nb
        self.scale = scale
        self.mean = mean

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.n_updates += self.nb
        with torch.no_grad():
            state = self.model.state_dict()
            for name in state:
                std, mean = torch.std_mean(state[name])
                state[name] = torch.rand(
                    state[name].size(), device=self.device) * self.scale + (self.mean - self.scale / 2.)
        self.model.load_state_dict(state)


class PegasosBackGradientAttackHandler(PegasosGradientScalingAttackHandler):
    def __init__(self,
                 nb: int,
                 net: AdaLine,
                 learning_rate: float,
                 create_model_mode: CreateModelMode = CreateModelMode.UPDATE,
                 copy_model: bool = True,
                 on_device: bool = False):
        PegasosGradientScalingAttackHandler.__init__(self,
                                                     -1., nb, net, learning_rate, create_model_mode, copy_model, on_device)
