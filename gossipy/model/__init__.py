import torch
from typing import Any

from torch.nn.modules.container import ParameterList
from .. import EqualityMixin, Sizeable
from ..utils import torch_models_eq

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["TorchModel"]


class TorchModel(torch.nn.Module, Sizeable):
    def __init__(self, *args, **kwargs):
        super(TorchModel, self).__init__()

    def init_weights(self, *args, **kwargs) -> None:
        raise NotImplementedError()
    
    def _get_n_params(self) -> int:
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    
    def get_size(self) -> int:
        return self._get_n_params()
    
    def __str__(self) -> str:
        return "TorchModel(size=%d)" %self.get_size()
    
    def get_params_list(self):
        return ParameterList(self.parameters())