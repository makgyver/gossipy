"""This module provides a series of classes to handle the models."""

from abc import ABC, abstractmethod
import torch
from torch.nn.modules.container import ParameterList

from .. import Sizeable

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache License, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["TorchModel"]


class TorchModel(torch.nn.Module, Sizeable, ABC):
    def __init__(self, *args, **kwargs):
        """Abstract class for a torch model.

        TorchModel is an abstract class that wraps a torch module and provide
        an interface to easily access the number of parameters of the module as well as
        to easily initialize the weights.
        """

        super(TorchModel, self).__init__()

    @abstractmethod
    def init_weights(self, *args, **kwargs) -> None:
        """Initialize the weights of the model."""

        pass
    
    def _get_n_params(self) -> int:
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    
    def get_size(self) -> int:
        """Returns the number of parameters of the model.
        
        Returns
        -------
        int
            The number of parameters of the model.
        """

        return self._get_n_params()
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return "%s(size=%d)" %(self.__class__.__name__, self.get_size())
    
    def get_params_list(self):
        """Returns a list of the parameters of the model as a torch.nn.ParameterList.

        Returns
        -------
        torch.nn.ParameterList
            A list of the parameters of the model.
        """

        return ParameterList(self.parameters())