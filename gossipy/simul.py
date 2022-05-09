from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.random import shuffle, random, randint, choice
from typing import Any, Callable, DefaultDict, Optional, Dict, List, Tuple
from rich.progress import track
import dill

from . import CACHE, AntiEntropyProtocol, LOG, CacheKey, Delay, set_seed
from .data import DataDispatcher
from .node import GossipNode
from .flow_control import TokenAccount
from .model.handler import ModelHandler
from .utils import plot_evaluation

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["GossipSimulator",
           "TokenizedGossipSimulator",
           "repeat_simulation"]

# TODO: implementing a simulation report class that summarize the statistics 
#       of the simulation, e.g., # sent message, failed message, size...


class SimulationEventReceiver(ABC):
    """
    The event receiver interface declares all the update methods, used by the event sender.
    """

    @abstractmethod
    def update_message(self, failed: bool, msg_size: Optional[int]=None) -> None:
        """
        Receive an update about a sent message or a failed message.

        Parameters
        ----------
        falied : bool
            Whether the message was sent or not.
        msg_size : int or None, default=None
            The size of the message.
        """
        pass

    def update_evaluation(self, round: int, on_user: bool, evaluation: List[Dict[str, float]]) -> None:
        """Receive an update about an evaluation.

        Parameters
        ----------
        round : int
            The round number.
        on_user : bool
            Whether the evaluation set is store on the clients/users or on the server.
        evaluation : list of dict[str, float]
            The evaluation metrics computed on each client.
        """
        pass

    def update_end(self) -> None:
        """Receive an update about the end of the simulation."""
        pass


class SimulationEventSender(ABC):
    """
    The event sender interface declares a set of methods for managing receviers.
    """

    _receivers: List[SimulationEventReceiver] = []

    def add_receiver(self, receiver: SimulationEventReceiver) -> None:
        """Attach an event receiver to the event sender.

        Parameters
        ----------
        receiver : SimulationEventReceiver
            The receiver to attach.
        """
        if receiver not in self._receivers:
            self._receivers.append(receiver)


    def remove_receiver(self, receiver: SimulationEventReceiver) -> None:
        """Detach an event receiver from the event sender.

        Parameters
        ----------
        receiver : SimulationEventReceiver
            The receiver to detach.
        """
        try:
            idx = self._receivers.index(receiver)
            self._receivers.pop(idx)
        except ValueError:
            pass


    def notify_message(self, falied: bool, msg_size: Optional[int]=None) -> None:
        """
        Notify all receivers about a sent message or a failed message.

        Parameters
        ----------
        falied : bool
            Whether the message was sent or not.
        msg_size : int or None, default=None
            The size of the message.
        """
        for er in self._receivers:
            er.update_message(falied, msg_size)


    def notify_evaluation(self, round: int, on_user:bool, evaluation: List[Dict[str, float]]) -> None:
        """Notify all receivers about an evaluation.   
        
        Parameters
        ----------
        round : int
            The round number.
        on_user : bool
            Whether the evaluation set is store on the clients/users or on the server.
        evaluation : list of dict[str, float]
            The evaluation metrics computed on each client.
        """
        for er in self._receivers:
            er.update_evaluation(round, on_user, evaluation)
    
    def notify_end(self) -> None:
        """Notify all receivers about the end of the simulation."""
        for er in self._receivers:
            er.update_end()


class SimulationReport(SimulationEventReceiver):
    def __init__(self):
        self.clear()
    
    def clear(self) -> None:
        """Clear the report."""
        self.sent_messages: int = 0
        self.total_size: int = 0
        self.failed_messages: int = 0
        self.global_evaluations: List[Tuple[int, Dict[str, float]]] = []
        self.local_evaluations: List[Tuple[int, Dict[str, float]]] = []
    
    def update_message(self, failed: bool, msg_size: Optional[int]=None) -> None:
        if failed:
            self.failed_messages += 1
        else:
            assert msg_size is not None, "msg_size is not set"
            self.sent_messages += 1
            self.total_size += msg_size
    
    def update_evaluation(self, round: int, on_user: bool, evaluation: List[Dict[str, float]]) -> None:
        ev = self._collect_results(evaluation)
        if on_user:
            self.local_evaluations.append((round, ev))
        else:
            self.global_evaluations.append((round, ev))
    
    def update_end(self) -> None:
        LOG.info("# Sent messages: %d" %self.sent_messages)
        LOG.info("# Failed messages: %d" %self.failed_messages)
        LOG.info("Total size: %d" %self.total_size)

    def _collect_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results: return {}
        res = {k: [] for k in results[0]}
        for k in res:
            for r in results:
                res[k].append(r[k])
            res[k] = np.mean(res[k])
        return res


class GossipSimulator(SimulationEventSender):
    def __init__(self,
                 data_dispatcher: DataDispatcher,
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 gossip_node_class: GossipNode,
                 gossip_node_params: Dict[str, Any],
                 model_handler_class: ModelHandler,
                 model_handler_params: Dict[str, Any],
                 topology: Optional[np.ndarray],
                 drop_prob: float=0., # [0,1] - probability of a message being dropped
                 online_prob: float=1., # [0,1] - probability of a node to be online
                 delay: Delay=Delay(0),
                 sampling_eval: float=0., # [0, 1] - percentage of nodes to evaluate
                 round_synced: bool=True):
        
        assert 0 <= drop_prob <= 1, "drop_prob must be in the range [0,1]."
        assert 0 <= online_prob <= 1, "online_prob must be in the range [0,1]."
        assert 0 <= sampling_eval <= 1, "sampling_eval must be in the range [0,1]."

        self.data_dispatcher = data_dispatcher
        self.n_nodes = data_dispatcher.size()
        self.delta = delta #round_len
        self.protocol = protocol
        self.drop_prob = drop_prob
        self.online_prob = online_prob
        self.delay = delay
        self.sampling_eval = sampling_eval
        self.gossip_node_class = gossip_node_class
        self.gossip_node_params = gossip_node_params
        self.model_handler_class = model_handler_class
        self.model_handler_params = model_handler_params
        self.topology = topology
        self.round_synced = round_synced
        self.initialized = False
        self.nodes = {}
        

    def init_nodes(self, seed:int=98765) -> None:
        self.initialized = True
        self.data_dispatcher.assign(seed)
        self.nodes = {i: self.gossip_node_class(idx=i,
                                                data=self.data_dispatcher[i],
                                                round_len=self.delta,
                                                n_nodes=self.n_nodes,
                                                model_handler=self.model_handler_class(**self.model_handler_params),
                                                known_nodes=self.topology[i] if self.topology is not None else None,
                                                sync=self.round_synced,
                                                **self.gossip_node_params)
                                                for i in range(self.n_nodes)}
        for _, node in self.nodes.items():
            node.init_model()
    
    # def add_nodes(self, nodes: List[GossipNode]) -> None:
    #     assert not self.initialized, "'init_nodes' must be called before adding new nodes."
    #     for node in nodes:
    #         node.idx = self.n_nodes
    #         node.init_model()
    #         self.nodes[node.idx] = node
    #         self.n_nodes += 1


    def start(self, n_rounds: int=100) -> None:
        assert self.initialized, "The simulator is not inizialized. Please, call the method 'init_nodes'."
        node_ids = np.arange(self.n_nodes)
        
        pbar = track(range(n_rounds * self.delta), description="Simulating...")
        msg_queues = DefaultDict(list)
        rep_queues = DefaultDict(list)

        try:
            for t in pbar:
                if t % self.delta == 0: shuffle(node_ids)
                
                for i in node_ids:
                    node = self.nodes[i]
                    if node.timed_out(t):
                        peer = node.get_peer()
                        msg = node.send(t, peer, self.protocol)
                        self.notify_message(False, msg.get_size())
                        if msg:
                            if random() >= self.drop_prob:
                                d = self.delay.get(msg)
                                msg_queues[t + d].append(msg)
                            else:
                                self.notify_message(True)
                
                for msg in msg_queues[t]:
                    if random() < self.online_prob:
                        reply = self.nodes[msg.receiver].receive(t, msg)
                        if reply:
                            if random() > self.drop_prob:
                                d = self.delay.get(reply)
                                rep_queues[t + d].append(reply)
                            else:
                                self.notify_message(True)
                del msg_queues[t]

                for reply in rep_queues[t]:
                    self.notify_message(False, reply.get_size())
                    self.nodes[reply.receiver].receive(t, reply)
                    
                del rep_queues[t]

                if (t+1) % self.delta == 0:
                    if self.sampling_eval > 0:
                        sample = choice(list(self.nodes.keys()), max(int(self.n_nodes * self.sampling_eval), 1))
                        ev = [self.nodes[i].evaluate() for i in sample if self.nodes[i].has_test()]
                    else:
                        ev = [n.evaluate() for _, n in self.nodes.items() if n.has_test()]
                    if ev:
                        self.notify_evaluation(t, True, ev)
                    
                    if self.data_dispatcher.has_test():
                        if self.sampling_eval > 0:
                            ev = [self.nodes[i].evaluate(self.data_dispatcher.get_eval_set())
                                for i in sample]
                        else:
                            ev = [n.evaluate(self.data_dispatcher.get_eval_set())
                                for _, n in self.nodes.items()]
                        if ev:
                            self.notify_evaluation(t, False, ev)

        except KeyboardInterrupt:
            LOG.warning("Simulation interrupted by user.")
        
        pbar.close()
        self.notify_end()
        return
    
    def save(self, filename) -> None:
        dump = {
            "simul": self,
            "cache": CACHE.get_cache()
        }
        with open(filename, 'wb') as f:
            dill.dump(dump, f)

    @classmethod
    def load(cls, filename) -> GossipSimulator:
        with open(filename, 'rb') as f:
            loaded = dill.load(f)
            CACHE.load(loaded["cache"])
            return loaded["simul"]
    
    def __repr__(self) -> str:
        pass


class TokenizedGossipSimulator(GossipSimulator):
    def __init__(self,
                 data_dispatcher: DataDispatcher,
                 token_account_class: TokenAccount,
                 token_account_params: Dict[str, Any],
                 utility_fun: Callable[[ModelHandler, ModelHandler], int],
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 gossip_node_class: GossipNode,
                 gossip_node_params: Dict[str, Any],
                 model_handler_class: ModelHandler,
                 model_handler_params: Dict[str, Any],
                 topology: Optional[np.ndarray],
                 drop_prob: float=0., # [0,1]
                 online_prob: float=1., # [0,1]
                 delay: Delay=Delay(0),
                 sampling_eval: float=0., #[0, 1]
                 round_synced: bool=True):
        super(TokenizedGossipSimulator, self).__init__(data_dispatcher,
                                                       delta,
                                                       protocol,
                                                       gossip_node_class,
                                                       gossip_node_params,
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
        pbar = track(range(n_rounds * self.delta), description="Simulating...")
        msg_queues = DefaultDict(list)
        rep_queues = DefaultDict(list)
        #avg_tokens = [0]
        try:
            for t in pbar:
                if t % self.delta == 0: 
                    shuffle(node_ids)
                    #if t > 0:
                    #    avg_tokens.append(np.mean([a.n_tokens for a in self.accounts.values()]))
                
                for i in node_ids:
                    node = self.nodes[i]
                    if node.timed_out(t):
                        if random() < self.accounts[i].proactive():
                            peer = node.get_peer()
                            msg = node.send(t, peer, self.protocol)
                            self.notify_message(False, msg.get_size())
                            if msg: 
                                if random() >= self.drop_prob:
                                    d = self.delay.get(msg)
                                    msg_queues[t + d].append(msg)
                                else:
                                    self.notify_message(True)
                        else:
                            self.accounts[i].add(1)

                for msg in msg_queues[t]:
                    if random() < self.online_prob:
                        if msg.value and isinstance(msg.value[0], CacheKey):
                            sender_mh = CACHE[msg.value[0]]
                        reply = self.nodes[msg.receiver].receive(t, msg)
                        if reply:
                            if random() > self.drop_prob:
                                d = self.delay.get(reply)
                                rep_queues[t + d].append(reply)
                            else:
                                self.notify_message(True)

                        if not reply:
                            utility = self.utility_fun(self.nodes[msg.receiver].model_handler, sender_mh)#msg.value[0])
                            reaction = self.accounts[msg.receiver].reactive(utility)
                            if reaction:
                                self.accounts[msg.receiver].sub(reaction)
                                for _ in range(reaction):
                                    peer = node.get_peer()
                                    msg = node.send(t, peer, self.protocol)
                                    self.notify_message(False, msg.get_size())
                                    if msg: 
                                        if random() >= self.drop_prob:
                                            d = self.delay.get(msg)
                                            msg_queues[t + d].append(msg)
                                        else:
                                            self.notify_message(True)
                del msg_queues[t]

                for reply in rep_queues[t]:
                    self.notify_message(False, reply.get_size())
                    self.nodes[reply.receiver].receive(t, reply)
                del rep_queues[t]

                if (t+1) % self.delta == 0:
                    if self.sampling_eval > 0:
                        sample = choice(list(self.nodes.keys()), max(int(self.n_nodes * self.sampling_eval), 1))
                        ev = [self.nodes[i].evaluate() for i in sample if self.nodes[i].has_test()]
                    else:
                        ev = [n.evaluate() for _, n in self.nodes.items() if n.has_test()]
                    if ev:
                        self.notify_evaluation(t, True, ev)
                    
                    if self.data_dispatcher.has_test():
                        if self.sampling_eval > 0:
                            ev = [self.nodes[i].evaluate(self.data_dispatcher.get_eval_set())
                                for i in sample]
                        else:
                            ev = [n.evaluate(self.data_dispatcher.get_eval_set())
                                for _, n in self.nodes.items()]
                        if ev:
                            self.notify_evaluation(t, False, ev)

        except KeyboardInterrupt:
            LOG.warning("Simulation interrupted by user.")

        pbar.close()
        self.notify_end()
        return


def repeat_simulation(gossip_simulator: GossipSimulator,
                      n_rounds: Optional[int]=1000,
                      repetitions: Optional[int]=10,
                      seed: int = 98765,
                      verbose: Optional[bool]=True) -> Tuple[List[List[float]], List[List[float]]]:
    
    report = SimulationReport()
    gossip_simulator.add_receiver(report)
    eval_list: List[List[float]] = []
    eval_user_list: List[List[float]] = []
    try:
        for i in range(1, repetitions+1):
            LOG.info("Simulation %d/%d" %(i, repetitions))
            gossip_simulator.init_nodes(seed*i)
            gossip_simulator.start(n_rounds=n_rounds)
            eval_list.append([ev for _, ev in report.global_evaluations])
            eval_user_list.append([ev for _, ev in report.local_evaluations])
            report.clear()
    except KeyboardInterrupt:
        LOG.info("Execution interrupted during the %d/%d simulation." %(i, repetitions))

    if verbose and eval_list:
        plot_evaluation(eval_list, "Overall test")
        plot_evaluation(eval_user_list, "User-wise test")
    
    return eval_list, eval_user_list
    