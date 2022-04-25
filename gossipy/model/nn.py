from collections import OrderedDict
import torch
from torch.nn import Module, Linear, Sequential
from torch.nn.init import xavier_uniform_
from torch.nn.modules.activation import ReLU, Sigmoid
from typing import Tuple
from . import TorchModel

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["TorchPerceptron", "TorchMLP", "AdaLine", "Pegasos", "LogisticRegression"]

class TorchPerceptron(TorchModel):
    def __init__(self, dim: int):
        super(TorchPerceptron, self).__init__()
        self.input_dim = dim
        self.model = Sequential(OrderedDict({
            "linear" : Linear(self.input_dim, 1), 
            "sigmoid" : Sigmoid()
        }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def init_weights(self) -> None:
        xavier_uniform_(self.model._modules['linear'].weight)
    
    def __repr__(self) -> str:
        return "TorchPerceptron(size=%d)\n%s" %(self.get_size(), str(self.model))


class TorchMLP(TorchModel):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Tuple[int]=(100,),
                 activation: Module=ReLU):
        super(TorchMLP, self).__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = OrderedDict()
        for i in range(len(dims)-1):
            layers["linear_%d" %(i+1)] = Linear(dims[i], dims[i+1])
            layers["activ_%d" %(i+1)] = activation()
        layers["linear_%d" %len(dims)] = Linear(dims[len(dims)-1], output_dim)
        #layers["softmax"] = Softmax(1)
        self.model = Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def init_weights(self) -> None:
        def _init_weights(m: Module):
            if type(m) == Linear:
                xavier_uniform_(m.weight)
        self.model.apply(_init_weights)
    
    def __repr__(self) -> str:
        return "TorchMLP(size=%d)\n%s" %(self.get_size(), str(self.model))


class AdaLine(TorchModel):
    def __init__(self, dim: int):
        super(AdaLine, self).__init__()
        self.input_dim = dim
        self.model = torch.nn.Parameter(torch.zeros(self.input_dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model @ x.T

    def get_size(self) -> int:
        return self.input_dim

    def init_weights(self) -> None:
        pass
    
    def __repr__(self) -> str:
        return "AdaLine(size=%d)" %(self.get_size())


class Pegasos(AdaLine):
    def __repr__(self) -> str:
        return "Pegasos(size=%d)" %(self.get_size())


class LogisticRegression(TorchModel):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def init_weights(self) -> None:
        pass
    
    def __repr__(self) -> str:
        return "LogisticRegression(in_size=%d, out_size=%d)" %(self.model.in_features,
                                                               self.model.out_features)
