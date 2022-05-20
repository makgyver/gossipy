import math
import torch
import numpy as np
from numpy.random import choice
from collections import Counter
from torch import LongTensor
from typing import Dict, Tuple, Optional
from torch.nn import ParameterList

from .. import LOG
from gossipy.model.nn import TorchModel

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["TorchModelSampling",
           "TorchModelPartition"]


class TorchModelSampling:
    """Class for sampling parameters from a torch model.
    
    This class only contains static methods because it does not need to know
    beforehand the specific type of model. It is therefore not possible to
    instantiate it.
    The sampling over a model is performed by randomly selecting a subset of its parameters.
    """


    @classmethod
    def sample(cls, size: float, net: TorchModel) -> Dict[int, Optional[Tuple[LongTensor, ...]]]:
        """Sample a subset of the parameters of a given model.

        Parameters
        ----------
        size : float
            The size (in percentage) of the subset to be sampled.
        net : TorchModel
            The model to be sampled.

        Returns
        -------
        Dict[int, Optional[Tuple[LongTensor, ...]]]
            A dictionary containing the indices of the parameters to be sampled. The keys are the indices of the
            layers, and the values are the indices of the parameters to be sampled in that layer.
        """
        assert 0 < size <= 1, "size must be in the range (0, 1]."
        if size >= 0.9:
            LOG.warning("You are using a high sample size (=%.2f) which can impact "\
                         "the performance without much advantage in terms of saved bandwith." %size)
        
        plist = ParameterList(net.parameters())
        probs = np.array([torch.numel(t) for t in plist], dtype='float')
        probs /= sum(probs)
        sample_size = max(1, int(round(size * net.get_size())))
        counter = dict(Counter(list(choice(len(plist), size=sample_size, p=probs))))
        samples = {i : None for i in range(len(plist))}
        for i, c in counter.items():
            tensor = plist[i]
            sizes = tuple(tensor.size())
            samples[i] = tuple([LongTensor(list(choice(s, size=c))) for s in sizes])
                
        return samples
    
    # FIXME: is this correct?
    @classmethod
    def merge(cls, sample: Dict[int, Optional[Tuple[LongTensor, ...]]],
                   net1: TorchModel,
                   net2: TorchModel,
                   reduce: str="mean") -> None:
        """Merge a sample of the parameters of two models.

        Parameters
        ----------
        sample : Dict[int, Optional[Tuple[LongTensor, ...]]]
            A dictionary containing the indices of the sampled parameters.
        net1 : TorchModel
            The first model.
        net2 : TorchModel
            The second model.
        reduce : {'mean', 'sum'}
            The reduction method to be used.
        """
        assert str(net1) == str(net2), "net1 and net2 must have the same architecture."
        assert reduce in {"mean", "sum"}, "reduce must be either 'sum' or 'mean'."

        plist1 = ParameterList(net1.parameters())
        plist2 = ParameterList(net2.parameters())

        assert len(plist1) == len(sample), "The provided sample is incompatible with the network."

        with torch.no_grad():
            for i in range(len(plist1)):
                t_ids = sample[i]
                if t_ids is not None:
                    mul = 2 if reduce == "mean" else 1
                    plist1[i][t_ids] = (plist1[i][t_ids] + plist2[i][t_ids]) / mul


class TorchModelPartition:
    def __init__(self, net_proto: TorchModel, n_parts: int):
        self._check(net_proto)
        self.str_arch = str(net_proto)
        self.n_parts = min(n_parts, net_proto.get_size())
        self.partitions = self._partition(net_proto, self.n_parts)
    
    def _check(self, net: TorchModel) -> None:
        plist = ParameterList(net.parameters())
        for t in plist:
            if t.dim() > 3:
                raise TypeError("Partitioning is only supported for neural "\
                                 "networks with at most 3D layers.")

    def _partition(self,
                   net: TorchModel,
                   n: int) -> Dict[int, Dict[int, Optional[Tuple[LongTensor, ...]]]]:
        plist = ParameterList(net.parameters())
        parts = {i : {j : None for j in range(len(plist))} for i in range(n)}
        net_size = net.get_size()
        mu = math.floor(net_size / n)
        rem = net_size % n
        ni, ti = 0, 0
        diff = mu + (rem > 0)
        shift = [0, 0, 0]
        ids = [[], [], []]
        while ti < len(plist):
            tensor = plist[ti]
            sizes = tuple(tensor.shape)
            cover = min(sizes[0] - shift[0], diff)
            diff -= cover

            ids[0].extend(range(shift[0], shift[0]+cover))
            if tensor.dim() >= 2: ids[1].extend([shift[1]] * cover)
            if tensor.dim() >= 3: ids[2].extend([shift[2]] * cover)

            shift[0] = (shift[0] + cover) % sizes[0]
            if not shift[0] and tensor.dim() >= 2: shift[1] = (shift[1] + 1) % sizes[1]
            if not shift[1] and tensor.dim() >= 3: shift[2] = (shift[2] + 1) % sizes[2]

            if tensor.dim() == 1:
                if diff == 0 or shift[0] == 0:
                    parts[ni][ti] = (torch.LongTensor(ids[0]),)
                    ids = [[], [], []]
            elif tensor.dim() == 2:
                if diff == 0 or shift[1] == 0:
                    parts[ni][ti] = (torch.LongTensor(ids[0]),
                                     torch.LongTensor(ids[1]))
                    ids = [[], [], []]
            else:#if tensor.dim() == 3:
                if diff == 0 or shift[2] == 0:
                    parts[ni][ti] = (torch.LongTensor(ids[0]),
                                     torch.LongTensor(ids[1]),
                                     torch.LongTensor(ids[2]))
                    ids = [[], [], []]
            
            if shift[0] == 0:
                if tensor.dim() == 1: ti += 1
                else:
                    if shift[1] == 0: 
                        if tensor.dim() == 2: ti += 1
                        elif shift[2] == 0: ti += 1

            if diff == 0:
                ni += 1
                diff = mu
                if ni < rem: diff += 1

        return parts
    

    def merge(self, id_part: int,
                    net1: TorchModel,
                    net2: TorchModel,
                    weights: Optional[Tuple[int, int]]=None) -> None:
        assert str(net1) == self.str_arch, "net1 is not compatible."
        assert str(net2) == self.str_arch, "net2 is not compatible."
        
        id_part = id_part % self.n_parts
        plist1 = ParameterList(net1.parameters())
        plist2 = ParameterList(net2.parameters())

        w = weights if (weights is not None and weights != (0,0)) else (1,1)
        mul1, mul2 = w[0] / sum(w), w[1] / sum(w)
        with torch.no_grad():
            for i in range(len(plist1)):
                t_ids = self.partitions[id_part][i]
                if t_ids is not None:
                    plist1[i][t_ids] = mul1 * plist1[i][t_ids] + mul2 * plist2[i][t_ids]
