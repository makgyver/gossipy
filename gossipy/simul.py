from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy.random import shuffle, random, choice
from typing import Callable, DefaultDict, Optional, Dict, List, Tuple
from rich.progress import track
import dill
import json

from . import CACHE, LOG, CacheKey
from .core import AntiEntropyProtocol, Message, ConstantDelay, Delay
from .data import DataDispatcher
from .node import GossipNode
from .flow_control import TokenAccount
from .model.handler import ModelHandler
from .utils import StringEncoder

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


__all__ = ["SimulationEventReceiver",
           "SimulationEventSender",
           "SimulationReport",
           "GossipSimulator",
           "TokenizedGossipSimulator"]


class SimulationEventReceiver(ABC):
    """The event receiver interface declares all the update methods, used by the event sender."""

    @abstractmethod
    def update_message(self, failed: bool, msg: Optional[Message]=None) -> None:
        """Receives an update about a sent or a failed message.

        Parameters
        ----------
        falied : bool
            Whether the message was sent (False) or not (True).
        msg_size : Message or None, default=None
            The message.
        """

        pass

    def update_evaluation(self,
                          round: int,
                          on_user: bool,
                          evaluation: List[Dict[str, float]]) -> None:
        """Receives an update about an evaluation.

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

    @abstractmethod
    def update_end(self) -> None:
        """Receives an update about the end of the simulation."""

        pass

    @abstractmethod
    def update_timestep(self, t: int):
        """Signals the end of the timestep ``t``.

        Parameters
        ----------
        t : int
            The current time step.
        """

        pass


class SimulationEventSender(ABC):
    """The event sender interface declares methods for managing receviers."""

    _receivers: List[SimulationEventReceiver] = []

    def add_receiver(self, receiver: SimulationEventReceiver) -> None:
        """Attaches an event receiver to the event sender.

        Parameters
        ----------
        receiver : SimulationEventReceiver
            The receiver to attach.
        """

        if receiver not in self._receivers:
            self._receivers.append(receiver)


    def remove_receiver(self, receiver: SimulationEventReceiver) -> None:
        """Detaches an event receiver from the event sender.

        If the ``receiver`` is not attached to the event sender, nothing happens.

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


    def notify_message(self, falied: bool, msg: Optional[Message]=None) -> None:
        """Notifies all receivers about a sent message or a failed message.

        Parameters
        ----------
        falied : bool
            Whether the message was sent or not.
        msg_size : Message or None, default=None
            The message.
        """

        for er in self._receivers:
            er.update_message(falied, msg)


    def notify_evaluation(self,
                          round: int,
                          on_user:bool,
                          evaluation: List[Dict[str, float]]) -> None:
        """Notifies all receivers about a performed evaluation.   
        
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
    
    def notify_timestep(self, t: int):
        """Notifies all receivers that a timestep has happened.
        
        Parameters
        ----------
        t : int
            The timestep number.
        """

        for er in self._receivers:
            er.update_timestep(t)

    def notify_end(self) -> None:
        """Notifies all receivers about the end of the simulation."""

        for er in self._receivers:
            er.update_end()


class SimulationReport(SimulationEventReceiver):
    
    _sent_messages: int
    _total_size: int
    _failed_messages: int
    _global_evaluations: List[Tuple[int, Dict[str, float]]]
    _local_evaluations: List[Tuple[int, Dict[str, float]]]

    def __init__(self):
        """Class that implements a basic simulation report.

        The report traces the number of sent messages, the number of failed messages,
        the total size of the sent messages, and the evaluation metrics (both global and local).

        The report is updated according to the design pattern Observer (actually Event Receiver).
        Thus, the report must be created and attached to the simulation before starting it.

        Examples
        --------
        >>> from gossipy.simul import SimulationReport
        >>> from gossipy.simul import GossipSimulator
        >>> simulator = GossipSimulator(...)
        >>> report = SimulationReport()
        >>> simulator.add_receiver(report)
        >>> simulator.start(...)

        The ``report`` object is now attached to the simulation and it will be notified about the
        events.

        See Also
        --------
        gossipy.Sizeable
        """

        self.clear()
    
    def clear(self) -> None:
        """Clears the report."""

        self._sent_messages = 0
        self._total_size = 0
        self._failed_messages = 0
        self._global_evaluations = []
        self._local_evaluations = []
    
    def update_message(self, failed: bool, msg: Optional[Message]=None) -> None:
        if failed:
            self._failed_messages += 1
        else:
            assert msg is not None, "msg is not set"
            self._sent_messages += 1
            self._total_size += msg.get_size()
    
    def update_evaluation(self,
                          round: int,
                          on_user: bool,
                          evaluation: List[Dict[str, float]]) -> None:
        ev = self._collect_results(evaluation)
        if on_user:
            self._local_evaluations.append((round, ev))
        else:
            self._global_evaluations.append((round, ev))
    
    def update_end(self) -> None:
        LOG.info("# Sent messages: %d" %self._sent_messages)
        LOG.info("# Failed messages: %d" %self._failed_messages)
        LOG.info("Total size: %d" %self._total_size)

    def _collect_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results: return {}
        res = {k: [] for k in results[0]}
        for k in res:
            for r in results:
                res[k].append(r[k])
            res[k] = np.mean(res[k])
        return res
    
    def get_evaluation(self, local: bool=False):
        if local:
            return self._local_evaluations
        else:
            return self._global_evaluations

    def update_timestep(self, t: int):
        pass


class GossipSimulator(SimulationEventSender):
    def __init__(self,
                 nodes: Dict[int, GossipNode],
                 data_dispatcher: DataDispatcher,
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 drop_prob: float=0., # [0,1] - probability of a message being dropped
                 online_prob: float=1., # [0,1] - probability of a node to be online
                 delay: Delay=ConstantDelay(0),
                 sampling_eval: float=0., # [0, 1] - percentage of nodes to evaluate
                ):
        """Class that implements a *vanilla* gossip learning simulation.

        TODO

        Parameters
        ----------
        nodes : dict[int, GossipNode]
            The nodes participating in the simulation. The keys are the node ids, and the values
            are the corresponding nodes (instances of the class :class:`GossipNode`).
        data_dispatcher : DataDispatcher
            The data dispatcher. Useful if the evaluation is performed on a separate test set, i.e.,
            not on the nodes.
        delta : int
            The number of timesteps of a round.
        protocol : AntiEntropyProtocol
            The protocol of the gossip simulation.
        drop_prob : float, default=0.
            The probability of a message being dropped.
        online_prob : float, default=1.
            The probability of a node to be online.
        delay : Delay, default=ConstantDelay(0)
            The (potential) function delay of the messages.
        sampling_eval : float, default=0.
            The percentage of nodes to use during evaluate. If 0 or 1, all nodes are considered.
        """
        
        assert 0 <= drop_prob <= 1, "drop_prob must be in the range [0,1]."
        assert 0 <= online_prob <= 1, "online_prob must be in the range [0,1]."
        assert 0 <= sampling_eval <= 1, "sampling_eval must be in the range [0,1]."

        self.data_dispatcher = data_dispatcher
        self.n_nodes = len(nodes)
        self.delta = delta #round_len
        self.protocol = protocol
        self.drop_prob = drop_prob
        self.online_prob = online_prob
        self.delay = delay
        self.sampling_eval = sampling_eval
        self.initialized = False
        self.nodes = nodes
        

    def init_nodes(self, seed:int=98765) -> None:
        """Initializes the nodes.

        The initialization of the nodes usually involves the initialization of the local model
        (see :meth:`GossipNode.init_model`).

        Parameters
        ----------
        seed : int, default=98765
            The seed for the random number generator.
        """

        self.initialized = True
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
        """Starts the simulation.

        The simulation handles the messages exchange between the nodes for ``n_rounds`` rounds.
        If attached to a :class:`SimulationReport`, the report is updated at each time step, 
        sent/fail message and evaluation.

        Parameters
        ----------
        n_rounds : int, default=100
            The number of rounds of the simulation.
        """

        assert self.initialized, \
               "The simulator is not inizialized. Please, call the method 'init_nodes'."
        LOG.info("Simulation started.")
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
                        self.notify_message(False, msg)
                        if msg:
                            if random() >= self.drop_prob:
                                d = self.delay.get(msg)
                                msg_queues[t + d].append(msg)
                            else:
                                self.notify_message(True)
                
                is_online = random(self.n_nodes) <= self.online_prob
                for msg in msg_queues[t]:
                    if is_online[msg.receiver]:
                        reply = self.nodes[msg.receiver].receive(t, msg)
                        if reply:
                            if random() > self.drop_prob:
                                d = self.delay.get(reply)
                                rep_queues[t + d].append(reply)
                            else:
                                self.notify_message(True)
                    else:
                        self.notify_message(True)
                del msg_queues[t]

                for reply in rep_queues[t]:
                    if is_online[reply.receiver]:
                        self.notify_message(False, reply)
                        self.nodes[reply.receiver].receive(t, reply)
                    else:
                        self.notify_message(True)
                    
                del rep_queues[t]

                if (t+1) % self.delta == 0:
                    if self.sampling_eval > 0:
                        sample = choice(list(self.nodes.keys()),
                                        max(int(self.n_nodes * self.sampling_eval), 1))
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
                self.notify_timestep(t)

        except KeyboardInterrupt:
            LOG.warning("Simulation interrupted by user.")
        
        pbar.close()
        self.notify_end()
        return
    
    def save(self, filename) -> None:
        """Saves the state of the simulator (including the models' cache).

        Parameters
        ----------
        filename : str
            The name of the file to save the state.
        """

        dump = {
            "simul": self,
            "cache": CACHE.get_cache()
        }
        with open(filename, 'wb') as f:
            dill.dump(dump, f)

    @classmethod
    def load(cls, filename) -> GossipSimulator:
        """Loads the state of the simulator (including the models' cache).

        Parameters
        ----------
        filename : str
            The name of the file to load the state.
        
        Returns
        -------
        GossipSimulator
            The simulator loaded from the file.
        """

        with open(filename, 'rb') as f:
            loaded = dill.load(f)
            CACHE.load(loaded["cache"])
            return loaded["simul"]
    
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        skip = ["nodes", "model_handler_params", "gossip_node_params"]
        attrs = {k: v for k, v in self.__dict__.items() if k not in skip}
        return f"{self.__class__.__name__} \
                 {str(json.dumps(attrs, indent=4, sort_keys=True, cls=StringEncoder))}"


class TokenizedGossipSimulator(GossipSimulator):
    def __init__(self,
                 nodes: Dict[int, GossipNode],
                 data_dispatcher: DataDispatcher,
                 token_account: TokenAccount,
                 utility_fun: Callable[[ModelHandler, ModelHandler], int],
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 drop_prob: float=0., # [0,1] - probability of a message being dropped
                 online_prob: float=1., # [0,1] - probability of a node to be online
                 delay: Delay=ConstantDelay(0),
                 sampling_eval: float=0., # [0, 1] - percentage of nodes to evaluate
                 ):
        super(TokenizedGossipSimulator, self).__init__(nodes,
                                                       data_dispatcher,
                                                       delta,
                                                       protocol,
                                                       drop_prob,
                                                       online_prob,
                                                       delay,
                                                       sampling_eval)
        self.utility_fun = utility_fun
        self.token_account_proto = token_account
        self.accounts = {}
    
    def init_nodes(self, seed: int=98765) -> None:
        super().init_nodes(seed)
        self.accounts = {i: deepcopy(self.token_account_proto) for i in range(self.n_nodes)}
        
    
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
                            self.notify_message(False, msg)
                            if msg: 
                                if random() >= self.drop_prob:
                                    d = self.delay.get(msg)
                                    msg_queues[t + d].append(msg)
                                else:
                                    self.notify_message(True)
                        else:
                            self.accounts[i].add(1)
                
                is_online = random(self.n_nodes) <= self.online_prob
                for msg in msg_queues[t]:
                    reply = None
                    if is_online[msg.receiver]:
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
                            utility = self.utility_fun(self.nodes[msg.receiver].model_handler,
                                                       sender_mh)#msg.value[0])
                            reaction = self.accounts[msg.receiver].reactive(utility)
                            if reaction:
                                self.accounts[msg.receiver].sub(reaction)
                                for _ in range(reaction):
                                    peer = node.get_peer()
                                    msg = node.send(t, peer, self.protocol)
                                    self.notify_message(False, msg)
                                    if msg: 
                                        if random() >= self.drop_prob:
                                            d = self.delay.get(msg)
                                            msg_queues[t + d].append(msg)
                                        else:
                                            self.notify_message(True)
                    else:
                        self.notify_message(True)
                
                del msg_queues[t]

                for reply in rep_queues[t]:
                    if is_online[reply.receiver]:
                        self.notify_message(False, reply)
                        self.nodes[reply.receiver].receive(t, reply)
                    else:
                        self.notify_message(True)
                del rep_queues[t]

                if (t+1) % self.delta == 0:
                    if self.sampling_eval > 0:
                        sample = choice(list(self.nodes.keys()),
                                        max(int(self.n_nodes * self.sampling_eval), 1))
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
                
                self.notify_timestep(t)

        except KeyboardInterrupt:
            LOG.warning("Simulation interrupted by user.")

        pbar.close()
        self.notify_end()
        return


# def repeat_simulation(gossip_simulator: GossipSimulator,
#                       n_rounds: Optional[int]=1000,
#                       repetitions: Optional[int]=10,
#                       seed: int=98765,
#                       verbose: Optional[bool]=True) -> Tuple[List[List[float]], List[List[float]]]:
    
#     report = SimulationReport()
#     gossip_simulator.add_receiver(report)
#     eval_list: List[List[float]] = []
#     eval_user_list: List[List[float]] = []
#     try:
#         for i in range(1, repetitions+1):
#             LOG.info("Simulation %d/%d" %(i, repetitions))
#             gossip_simulator.init_nodes(seed*i)
#             gossip_simulator.start(n_rounds=n_rounds)
#             eval_list.append([ev for _, ev in report.get_evaluation(False)])
#             eval_user_list.append([ev for _, ev in report.get_evaluation(True)])
#             report.clear()
#     except KeyboardInterrupt:
#         LOG.info("Execution interrupted during the %d/%d simulation." %(i, repetitions))

#     if verbose and eval_list:
#         plot_evaluation(eval_list, "Overall test")
#         plot_evaluation(eval_user_list, "User-wise test")
    
#     return eval_list, eval_user_list
    