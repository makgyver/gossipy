import torch
import pandas as pd
import numpy as np
from numpy.random import choice
from typing import OrderedDict, Union
from torch.nn import functional as F
from baseline import sklearn_mlp, torch_mlp
from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode, UAGossipNode, MABGossipNode
from gossipy.model.handler import TorchModelHandler
from gossipy.model.nn import TorchMLP
from gossipy.data import load_classification_dataset, DataDispatcher, DataHandler
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, repeat_simulation
from gossipy.utils import print_flush

def load_marvel_dataset(with_sensitive: bool=True, as_tensor: bool=True):
    df = pd.read_csv("datasets/marvel_with%s_sensible.csv" %("out" if not with_sensitive else ""))
    df = df.drop("ID", axis=1)
    X = df[df.columns[df.columns != "Target"]].to_numpy()
    y = df["Target"].to_numpy()
    if as_tensor:
        X = torch.tensor(X).float()
        y = torch.tensor(y).float().reshape(y.shape[0], 1)
    return X, y, df.columns.get_loc("SEX_Female Characters"), df.columns.get_loc("SEX_Male Characters")


class MarvelDataHandler(ClassificationDataHandler):
    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 feat: int,
                 test_size: float=0.2,
                 seed: int=42,
                 remove_feats=False):
        super(MarvelDataHandler, self).__init__(X, y, test_size, seed)
        assert(0 < test_size < 1)
        values = set(X[:, feat])
        self.feat = feat
        self.tr_fmap = OrderedDict({v : (self.Xtr[:, feat] == v).nonzero()[0] for v in values})
        self.te_fmap = OrderedDict({v : (self.Xte[:, feat] == v).nonzero()[0] for v in values})

        if remove_feats:
            self.Xtr = np.delete(self.Xtr, np.s_[feat:feat+2], axis=1)
            self.Xte = np.delete(self.Xte, np.s_[feat:feat+2], axis=1)
        
    def to_tensor(self):
        self.Xtr = torch.tensor(self.Xtr).float()
        self.ytr = torch.tensor(self.ytr).float().reshape(self.ytr.shape[0], 1)
        if self.Xte is not None:
            self.Xte = torch.tensor(self.Xte).float()
            self.yte = torch.tensor(self.yte).float().reshape(self.yte.shape[0], 1)
        

class MarvelDataDispatcher(DataDispatcher):
    def __init__(self,
                 data_handler: MarvelDataHandler,
                 n: int=0,
                 conn_perc: float=0.3,
                 eval_on_user: bool=True):
        assert(data_handler.size() >= n)
        if n <= 1: n = data_handler.size()
        self.data_handler = data_handler
        self.n = n
        self.eval_on_user = eval_on_user
        self.tr_assignments = [[] for _ in range(n)]
        self.te_assignments = [[] for _ in range(n)]

        tr_perc = {v : len(data_handler.tr_fmap[v]) / data_handler.size() for v in data_handler.tr_fmap}

        clients_x_v = OrderedDict()
        v = tr_cnt = 0
        for v in data_handler.tr_fmap:
            clients_x_v[v] = round(n * tr_perc[v])
            tr_cnt += clients_x_v[v]
        clients_x_v[v] += (n - tr_cnt)

        shift = 0
        for v in data_handler.tr_fmap:
            for i, iv in enumerate(data_handler.tr_fmap[v]):
                self.tr_assignments[shift + (i % clients_x_v[v])].append(iv)
            shift += clients_x_v[v]
        
        if eval_on_user:
            shift = 0
            for v in data_handler.te_fmap:
                for i, iv in enumerate(data_handler.te_fmap[v]):
                    self.te_assignments[shift + (i % clients_x_v[v])].append(iv)
                shift += clients_x_v[v]

        topology = np.zeros((n, n))
        shift = 0
        for v in clients_x_v:
            c = list(range(shift, shift + clients_x_v[v]))
            topology[np.ix_(c, c)] = 1
            shift += clients_x_v[v]
        np.fill_diagonal(topology, 0)

        shift1 = 0
        for i, v1 in enumerate(clients_x_v):
            shift2 = 0
            for j, v2 in enumerate(clients_x_v):
                if j > i:
                    pmax = clients_x_v[v1] * clients_x_v[v2]
                    k =  int(pmax*conn_perc)
                    cnt = 0
                    while cnt < k:
                        k1 = choice(range(shift1, shift1+clients_x_v[v1]))
                        k2 = choice(range(shift2, shift2+clients_x_v[v2]))
                        if topology[k1, k2] != 1:
                            topology[k1, k2] = topology[k2, k1] = 1
                            cnt += 1
                shift2 += clients_x_v[v2]
            shift1 += clients_x_v[v1]
        
        self.topology = topology


set_seed(98765)
X, y, f, m = load_marvel_dataset(True, as_tensor=False)
data_handler = MarvelDataHandler(X, y, f, test_size=.3, remove_feats=False)
dispatcher = MarvelDataDispatcher(data_handler, n=100, conn_perc=0.0, eval_on_user=True)
data_handler.to_tensor()

# print_flush("PyTorch MLP acc - T:%.4f, F:%.4f, M:%.4f" %torch_mlp(data_handler, verbose=False))
#print_flush("Sklearn MLP acc - T:%.4f, F:%.4f, M:%.4f" %sklearn_mlp(data_handler, verbose=False))
# X, y, f, m = load_marvel_dataset(True, as_tensor=True)
# data_handler = ClassificationDataHandler(X, y, test_size=.3)
# dispatcher = DataDispatcher(data_handler, n=100, eval_on_user=False)

#topology_fun = lambda: dispatcher.topology
topology_fun = None

set_seed(98765)
res = repeat_simulation(data_dispatcher=dispatcher,
                        round_delta=100,
                        protocol=AntiEntropyProtocol.PUSH, 
                        gossip_node=GossipNode,
                        model_handler=TorchModelHandler,
                        model_handler_params={"net" : TorchMLP(data_handler.Xtr.shape[1]), #TorchMLP
                                              "optimizer" : torch.optim.SGD,
                                              "l2_reg": 0.001,
                                              "criterion" : F.mse_loss,
                                              "learning_rate" : .1,
                                              "create_model_mode" : CreateModelMode.UPDATE_MERGE},
                        topology_fun=topology_fun,
                        n_iter=1000,
                        repetitions=5,
                        round_synced=True,
                        verbose=True)


# mu1 = np.mean(res[1], axis=0)
# mu2 = np.mean(res2[1], axis=0)
# plt.plot(range(len(mu1)), mu1)
# plt.plot(range(len(mu2)), mu2)
# plt.show()
