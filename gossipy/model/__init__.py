import torch
from .. import Sizeable

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