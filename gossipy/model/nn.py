import torch
from . import TorchModel

__all__ = ["TorchPerceptron", "TorchMLP"]

class TorchPerceptron(TorchModel):
    def __init__(self, dim: int):
        super(TorchPerceptron, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w(x))

    def init_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.w.weight)
    
    def __str__(self) -> str:
        return "TorchPerceptron(size=%d)" %self.get_size()

#FIXME: generalize to n hidden layers
class TorchMLP(TorchModel):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int=100):
        super(TorchMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

    def init_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    
    def __str__(self) -> str:
        return "TorchMLP(size=%d)" %self.get_size()