import math
import torch
from torch.nn import ParameterList

from .. import LOG
from gossipy.model.nn import TorchModel, TorchPerceptron

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


class TorchPartitionManager:
    def __init__(self, net_proto: TorchModel, n_parts: int):
        self._check(net_proto)
        self.str_arch = str(net_proto)
        self.n_parts = min(n_parts, net_proto.get_size())
        self.partitions = self._partition(net_proto, self.n_parts)
    
    def _check(self, net: TorchModel):
        plist = ParameterList(net.parameters())
        for t in plist:
            if t.dim() > 3:
                raise TypeError("Partitioning is only not supported on\
                                 networks with at most 3D layers.")

    def _partition(self, net: TorchModel, n: int):
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
    
    def is_compatible(self, net: TorchModel):
        return self.str_arch == str(net)
    
    def __getitem__(self, idx: int):
        return [[dim[idx] for dim in params] for params in self.partitions]

    def merge(self, id_part: int,
                    net1: TorchModel,
                    net2: TorchModel,
                    reduce: str="mean"):
        assert self.is_compatible(net1), "net1 is not compatible."
        assert self.is_compatible(net2), "net2 is not compatible."
        assert reduce in {"mean", "sum"}, "reduce can be either 'sum' or 'mean'."

        if id_part >= self.n_parts:
            LOG.warning("Skipped merging models on non existing partition id.")
            return

        plist1 = ParameterList(net1.parameters())
        plist2 = ParameterList(net2.parameters())

        with torch.no_grad():
            for i in range(len(plist1)):
                t_ids = self.partitions[id_part][i]
                if t_ids is not None:
                    mul = 2 if reduce == "mean" else 1
                    plist1[i][t_ids] = (plist1[i][t_ids] + plist2[i][t_ids]) * mul
                    

                





