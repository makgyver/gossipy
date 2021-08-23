from __future__ import annotations
import numpy as np
from numpy.random import shuffle, random
from typing import Any, Optional, Dict, List, Callable, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from . import AntiEntropyProtocol
from .data import DataDispatcher, DataHandler
from .node import GossipNode
from .utils import print_flush
from .model.handler import ModelHandler

__all__ = ["GossipSimulator", "repeat_simulation"]

#FIXME: fix all the evaluation part
class GossipSimulator():
    def __init__(self,
                 data_dispatcher: DataDispatcher,
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 gossip_node: GossipNode,
                 model_handler: ModelHandler,
                 model_handler_params: Dict[str, Any],
                 topology: Optional[np.ndarray],
                 message_failure_rate: float=0., # [0,1]
                 online_prob: float=1., # [0,1]
                 round_synced: bool=True): 

        self.data_dispatcher = data_dispatcher
        self.n_nodes = data_dispatcher.size()
        self.delta = delta
        self.protocol = protocol
        self.topology = topology
        self.message_failure_rate = message_failure_rate
        self.online_prob = online_prob
        self.nodes = {i: gossip_node(i,
                                     data_dispatcher[i],
                                     delta,
                                     self.n_nodes,
                                     model_handler(**model_handler_params),
                                     topology[i] if topology is not None else None,
                                     round_synced)
                                     for i in range(self.n_nodes)}

    def _init_nodes(self) -> None:
        for _, node in self.nodes.items():
            node.init_model(True)

    def _collect_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        res = {k: [] for k in results[0]}
        for k in res:
            for r in results:
                res[k].append(r[k])
            res[k] = np.mean(res[k])
        return res

    def start(self, n_iter: int=10_000, scratch=True) -> List[float]:
        if scratch: self._init_nodes()
        node_ids = np.arange(self.n_nodes)
        pbar = tqdm(range(n_iter))
        evals = []
        evals_user = []
        n_mgs = 0
        tot_size = 0
        for t in pbar:
            if t % self.delta == 0: shuffle(node_ids)
            
            message_queue = []
            for i in node_ids:
                node = self.nodes[i]
                if node.timed_out(t):
                    msg = node.send(t, self.protocol)
                    n_mgs += 1
                    tot_size += msg.get_size()
                    if msg and random() > self.message_failure_rate:
                        message_queue.append(msg)
            
            reply_queue = []
            for msg in message_queue:
                if random() < self.online_prob:
                    reply = self.nodes[msg.receiver].receive(msg)
                    if reply and random() > self.message_failure_rate:
                        reply_queue.append(reply)
            
            for reply in reply_queue:
                self.nodes[reply.receiver].receive(reply)
                n_mgs += 1
                tot_size += reply.get_size()

            evaluation = 0
            if (t+1) % self.delta == 0:
                evaluation_user = self._collect_results([n.evaluate() for _, n in self.nodes.items() if n.has_test()])
                evaluation = self._collect_results([n.evaluate(self.data_dispatcher.get_eval_set()) for _, n in self.nodes.items()])

                evals.append(evaluation)
                evals_user.append(evaluation_user)

        print("# Mgs:    ", n_mgs)
        print("Tot. size:", tot_size)
        return evals, evals_user #, n_mgs, tot_size
    
    def save(self, filename) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename) -> GossipSimulator:
        with open(filename, 'rb') as f:
            return pickle.load(f)



def plot_evaluation(evals, title):
    for k in evals[0][0]:
        evs = [[d[k] for d in l] for l in evals]
        mu: float = np.mean(evs, axis=0)
        std: float = np.std(evs, axis=0)
        plt.plot(range(len(mu)), mu, label=k)
        plt.fill_between(range(len(mu)), mu-std, mu+std, alpha=0.2)
        plt.legend(loc="lower right")
        plt.title(title)
    plt.show()

def repeat_simulation(data_dispatcher: DataDispatcher,
                      round_delta: int,
                      protocol: AntiEntropyProtocol,
                      gossip_node: GossipNode,
                      model_handler: ModelHandler,
                      model_handler_params: Dict[str, Any],
                      topology_fun: Optional[Callable[[], np.ndarray]],
                      n_iter: Optional[int]=1000,
                      repetitions: Optional[int]=10,
                      round_synced: bool=True,
                      verbose: Optional[bool]=True) -> Tuple[List[GossipSimulator],
                                                             List[List[float]]]:
    
    eval_list: List[List[float]] = []
    eval_user_list: List[List[float]] = []
    sims: List[GossipSimulator] = [None for i in range(repetitions)]
    try:
        for i in range(repetitions):
            print_flush("Simulation %d/%d" %(i+1, repetitions))
            topology = topology_fun() if topology_fun is not None else None
            sims[i] = GossipSimulator(data_dispatcher=data_dispatcher,
                                      delta=round_delta,
                                      protocol=protocol,
                                      gossip_node=gossip_node,
                                      model_handler=model_handler,
                                      model_handler_params=model_handler_params,
                                      topology=topology,
                                      message_failure_rate=0,
                                      online_prob=1,
                                      round_synced=round_synced)
            evaluation, evaluation_user = sims[i].start(n_iter=n_iter)
            eval_list.append(evaluation)
            eval_user_list.append(evaluation_user)
    except KeyboardInterrupt:
        print_flush("Execution interrupted during the %d/%d simulation." %(i+1, repetitions))

    #print(np.mean([n.n_updates for _,n in sims[0].nodes.items()]))
    #print(np.min([n.n_updates for _,n in sims[0].nodes.items()]))
    #print(np.max([n.n_updates for _,n in sims[0].nodes.items()]))
    if verbose and eval_list:
        plot_evaluation(eval_list, "Overall test")
        plot_evaluation(eval_user_list, "User-wise test")
    
    return sims, eval_list, eval_user_list