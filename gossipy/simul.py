from __future__ import annotations
import numpy as np
from numpy.lib.arraysetops import isin
from numpy.random import shuffle, random, randint, choice
from typing import Any, Callable, DefaultDict, Optional, Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import dill

from . import AntiEntropyProtocol, LOG, CacheKey
from .data import DataDispatcher
from .node import GossipNode, TokenAccount
from .model.handler import ModelHandler

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["GossipSimulator", "plot_evaluation", "repeat_simulation"]


class GossipSimulator():
    def __init__(self,
                 data_dispatcher: DataDispatcher,
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 gossip_node_class: GossipNode,
                 model_handler_class: ModelHandler,
                 model_handler_params: Dict[str, Any],
                 topology: Optional[np.ndarray],
                 drop_prob: float=0., # [0,1]
                 online_prob: float=1., # [0,1]
                 delay: Optional[Tuple[int, int]]=None,
                 sampling_eval: float=0., #[0, 1]
                 round_synced: bool=True):
        
        assert 0 <= drop_prob <= 1, "drop_prob must be in the range [0,1]."
        assert 0 <= online_prob <= 1, "online_prob must be in the range [0,1]."
        assert 0 <= sampling_eval <= 1, "sampling_eval must be in the range [0,1]."
        assert (not delay) or (0 <= delay[0] <= delay[1]), "delay value is not correct."

        self.data_dispatcher = data_dispatcher
        self.n_nodes = data_dispatcher.size()
        self.delta = delta #round_len
        self.protocol = protocol
        self.topology = topology
        self.drop_prob = drop_prob
        self.online_prob = online_prob
        self.delay = delay
        self.sampling_eval = sampling_eval
        self.gossip_node_class = gossip_node_class
        self.model_handler_class = model_handler_class
        self.model_handler_params = model_handler_params
        self.topology = topology
        self.round_synced = round_synced
        self.initialized = False
        

    def init_nodes(self, seed:int=98765) -> None:
        self.initialized = True
        self.data_dispatcher.assign(seed)
        self.nodes = {i: self.gossip_node_class(i,
                                           self.data_dispatcher[i],
                                           self.delta,
                                           self.n_nodes,
                                           self.model_handler_class(**self.model_handler_params),
                                           self.topology[i] if self.topology is not None else None,
                                           self.round_synced)
                                           for i in range(self.n_nodes)}
        for _, node in self.nodes.items():
            node.init_model()

    def _collect_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results: return {}
        res = {k: [] for k in results[0]}
        for k in res:
            for r in results:
                res[k].append(r[k])
            res[k] = np.mean(res[k])
        return res

    def start(self, n_rounds: int=100) -> Tuple[List[float], List[float]]:
        assert self.initialized, "The simulator is not inizialized. Please, call the method 'init_nodes'."
        node_ids = np.arange(self.n_nodes)
        pbar = tqdm(range(n_rounds * self.delta))
        evals = []
        evals_user = []
        n_msg = 0
        n_msg_failed = 0
        tot_size = 0
        msg_queues = DefaultDict(list)
        rep_queues = DefaultDict(list)
        for t in pbar:
            if t % self.delta == 0: shuffle(node_ids)
            
            for i in node_ids:
                node = self.nodes[i]
                if node.timed_out(t):
                    peer = node.get_peer()
                    msg = node.send(t, peer, self.protocol)
                    n_msg += 1
                    tot_size += msg.get_size()
                    if msg: 
                        if random() >= self.drop_prob:
                            d = randint(self.delay[0], self.delay[1]+1) if self.delay else 0
                            msg_queues[t + d].append(msg)
                        else:
                            n_msg_failed += 1
            
            for msg in msg_queues[t]:
                if random() < self.online_prob:
                    reply = self.nodes[msg.receiver].receive(t, msg)
                    if reply:
                        if random() > self.drop_prob:
                            d = randint(self.delay[0], self.delay[1]+1) if self.delay else 0
                            rep_queues[t + d].append(reply)
                        else:
                            n_msg_failed += 1
            del msg_queues[t]

            for reply in rep_queues[t]:
                tot_size += reply.get_size()
                self.nodes[reply.receiver].receive(t, reply)
                n_msg += 1
            del rep_queues[t]

            if (t+1) % self.delta == 0:
                if self.sampling_eval > 0:
                    sample = choice(list(self.nodes.keys()), int(self.n_nodes * self.sampling_eval))
                    ev = [self.nodes[i].evaluate() for i in sample if self.nodes[i].has_test()]
                else:
                    ev = [n.evaluate() for _, n in self.nodes.items() if n.has_test()]
                evals_user.append(self._collect_results(ev))
                
                if self.data_dispatcher.has_test():
                    if self.sampling_eval > 0:
                        ev = [self.nodes[i].evaluate(self.data_dispatcher.get_eval_set())
                              for i in sample]
                    else:
                        ev = [n.evaluate(self.data_dispatcher.get_eval_set())
                              for _, n in self.nodes.items()]
                    evals.append(self._collect_results(ev))

        LOG.info("# Sent messages: %d" %n_msg)
        LOG.info("# Failed messages: %d" %n_msg_failed)
        LOG.info("Total size: %d" %tot_size)
        return evals, evals_user
    
    def save(self, filename) -> None:
        dump = {
            "simul": self,
            "cache": ModelHandler.cache
        }
        with open(filename, 'wb') as f:
            dill.dump(dump, f)

    @classmethod
    def load(cls, filename) -> GossipSimulator:
        with open(filename, 'rb') as f:
            loaded = dill.load(f)
            ModelHandler.cache = loaded["cache"]
            return loaded["simul"]


class TokenizedGossipSimulator(GossipSimulator):
    def __init__(self,
                 data_dispatcher: DataDispatcher,
                 token_account_class: TokenAccount,
                 token_account_params: Dict[str, Any],
                 utility_fun: Callable[[ModelHandler, ModelHandler], int],
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 gossip_node_class: GossipNode,
                 model_handler_class: ModelHandler,
                 model_handler_params: Dict[str, Any],
                 topology: Optional[np.ndarray],
                 drop_prob: float=0., # [0,1]
                 online_prob: float=1., # [0,1]
                 delay: Optional[Tuple[int, int]]=None,
                 sampling_eval: float=0., #[0, 1]
                 round_synced: bool=True):
        super(TokenizedGossipSimulator, self).__init__(data_dispatcher,
                                                       delta,
                                                       protocol,
                                                       gossip_node_class,
                                                       model_handler_class,
                                                       model_handler_params,
                                                       topology,
                                                       drop_prob,
                                                       online_prob,
                                                       delay,
                                                       sampling_eval,
                                                       round_synced)
        self.utility_fun = utility_fun
        self.token_account_class = token_account_class
        self.token_account_params = token_account_params
    
    def init_nodes(self, seed: int=98765) -> None:
        super().init_nodes(seed)
        self.accounts = {i: self.token_account_class(**self.token_account_params)
                            for i in range(self.n_nodes)}
    
    def start(self, n_rounds: int=100) -> Tuple[List[float], List[float]]:
        node_ids = np.arange(self.n_nodes)
        pbar = tqdm(range(n_rounds * self.delta))
        evals = []
        evals_user = []
        n_msg = 0
        n_msg_failed = 0
        tot_size = 0
        msg_queues = DefaultDict(list)
        rep_queues = DefaultDict(list)
        for t in pbar:
            if t % self.delta == 0: shuffle(node_ids)
            
            for i in node_ids:
                node = self.nodes[i]
                if node.timed_out(t):
                    if random() < self.accounts[i].proactive():
                        peer = node.get_peer()
                        msg = node.send(t, peer, self.protocol)
                        n_msg += 1
                        tot_size += msg.get_size()
                        if msg: 
                            if random() >= self.drop_prob:
                                d = randint(self.delay[0], self.delay[1]+1) if self.delay else 0
                                msg_queues[t + d].append(msg)
                            else:
                                n_msg_failed += 1
                    else:
                        self.accounts[i].add(1)

            for msg in msg_queues[t]:
                if random() < self.online_prob:
                    if msg.value and isinstance(msg.value[0], CacheKey):
                        sender_mh = ModelHandler.cache[msg.value[0]].value
                    reply = self.nodes[msg.receiver].receive(t, msg)
                    if reply:
                        if random() > self.drop_prob:
                            d = randint(self.delay[0], self.delay[1]+1) if self.delay else 0
                            rep_queues[t + d].append(reply)
                        else:
                            n_msg_failed += 1

                    if not reply:
                        utility = self.utility_fun(self.nodes[msg.receiver].model_handler, 
                                                   sender_mh)#msg.value[0])
                        reaction = self.accounts[msg.receiver].reactive(utility)
                        if reaction:
                            self.accounts[msg.receiver].sub(reaction)
                            for _ in range(reaction):
                                peer = node.get_peer()
                                msg = node.send(t, peer, self.protocol)
                                n_msg += 1
                                tot_size += msg.get_size()
                                if msg: 
                                    if random() >= self.drop_prob:
                                        d = randint(self.delay[0], self.delay[1]+1) if self.delay else 1
                                        msg_queues[t + d].append(msg)
                                    else:
                                        n_msg_failed += 1
            del msg_queues[t]

            for reply in rep_queues[t]:
                tot_size += reply.get_size()
                self.nodes[reply.receiver].receive(t, reply)
                n_msg += 1
            del rep_queues[t]

            if (t+1) % self.delta == 0:
                if self.sampling_eval > 0:
                    sample = choice(list(self.nodes.keys()), int(self.n_nodes * self.sampling_eval))
                    ev = [self.nodes[i].evaluate() for i in sample if self.nodes[i].has_test()]
                else:
                    ev = [n.evaluate() for _, n in self.nodes.items() if n.has_test()]
                evals_user.append(self._collect_results(ev))
                
                if self.data_dispatcher.has_test():
                    if self.sampling_eval > 0:
                        ev = [self.nodes[i].evaluate(self.data_dispatcher.get_eval_set())
                              for i in sample]
                    else:
                        ev = [n.evaluate(self.data_dispatcher.get_eval_set())
                              for _, n in self.nodes.items()]
                    evals.append(self._collect_results(ev))

        LOG.info("# Sent messages: %d" %n_msg)
        LOG.info("# Failed messages: %d" %n_msg_failed)
        LOG.info("Total size: %d" %tot_size)
        return evals, evals_user


def plot_evaluation(evals: List[List[Dict]],
                    title: str="No title") -> None:
    if not evals[0][0]: return
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    for k in evals[0][0]:
        evs = [[d[k] for d in l] for l in evals]
        mu: float = np.mean(evs, axis=0)
        std: float = np.std(evs, axis=0)
        plt.fill_between(range(1, len(mu)+1), mu-std, mu+std, alpha=0.2)
        plt.title(title)
        plt.xlabel("cycle")
        plt.ylabel("value")
        plt.plot(range(1, len(mu)+1), mu, label=k)
    ax.legend(loc="lower right")
    plt.show()


def repeat_simulation(gossip_simulator: GossipSimulator,
                      n_rounds: Optional[int]=1000,
                      repetitions: Optional[int]=10,
                      seed: int = 98765,
                      verbose: Optional[bool]=True) -> Tuple[List[List[float]], List[List[float]]]:
    
    eval_list: List[List[float]] = []
    eval_user_list: List[List[float]] = []
    try:
        for i in range(repetitions):
            LOG.info("Simulation %d/%d" %(i+1, repetitions))
            gossip_simulator.init_nodes(98765*i)
            evaluation, evaluation_user = gossip_simulator.start(n_rounds=n_rounds)
            eval_list.append(evaluation)
            eval_user_list.append(evaluation_user)
    except KeyboardInterrupt:
        LOG.info("Execution interrupted during the %d/%d simulation." %(i+1, repetitions))

    if verbose and eval_list:
        plot_evaluation(eval_list, "Overall test")
        plot_evaluation(eval_user_list, "User-wise test")
    
    return eval_list, eval_user_list