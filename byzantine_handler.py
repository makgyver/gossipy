from __future__ import annotations
from typing import Tuple
import torch
from gossipy.model.handler import ModelHandler, TorchModelHandler

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
    "RandomAttackMixin",
    "SameValueAttackMixin",
    "GradientScalingAttackMixin",
    "BackGradientAttackMixin"
]


class AvoidReportMixin():
    '''Inheriting objects won't be taken into account in the evaluation when using compatible SimulationReport.
    Usefull to avoir malicious clients voluntarily impacting results.'''

    def evaluate(self, *args, **kwargs):
        return None


class RandomAttackMixin(AvoidReportMixin, TorchModelHandler):
    def __init__(self, noise: float):
        '''Adds a random vector to the model at each update.
        Acts like a getting a random gradient at each update.

        Parameters
        ----------
        noise: float
            The amplitude of noise
        '''
        self.noise = noise

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn(param.size()) * self.noise)


class SameValueAttackMixin(AvoidReportMixin, ModelHandler):
    '''Does nothing at each update.
    Acts like a getting a null gradient at each update.
    '''

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        pass


class GradientScalingAttackMixin(AvoidReportMixin, TorchModelHandler):
    def __init__(self, scale: float):
        '''Scales the gradient of each update by a factor

        Parameters
        ----------
        scale: float
            The scale used.
        '''
        self.scale = scale

    def _update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = data
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
        super(BackGradientAttackMixin, self).__init__(-1)
