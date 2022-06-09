from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy.random import shuffle, random, choice
from typing import Callable, DefaultDict, Optional, Dict, List, Tuple
from rich.progress import track
import dill
import json

from gossipy import CACHE, LOG, CacheKey
from gossipy.core import AntiEntropyProtocol, Message, ConstantDelay, Delay
from gossipy.data import DataDispatcher
from gossipy.node import GossipNode
from gossipy.flow_control import TokenAccount
from gossipy.model.handler import ModelHandler
from gossipy.utils import StringEncoder
from gossipy.simul import GossipSimulator, TokenizedGossipSimulator

from byzantine_handler import AvoidReportMixin


# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = [
    "GossipSimulatorByzantine"
]


class GossipSimulatorByzantine(GossipSimulator):
    def __init__(self,
                 nodes: Dict[int, GossipNode],
                 data_dispatcher: DataDispatcher,
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 drop_prob: float = 0.,
                 online_prob: float = 1.,
                 delay: Delay = ConstantDelay(0),
                 sampling_eval: float = 0.,
                 ):
        """Same as the GossipSimulator class but removes Byzantine attack model handler from evaluation.
        Evaluation is still done but the result is not used"""
        super().__init__(nodes, data_dispatcher, delta,
                         protocol, drop_prob, online_prob, delay, sampling_eval)
        self.not_byzantine_indices = [i for i in range(self.n_nodes) if not isinstance(
            self.nodes[i].model_handler, AvoidReportMixin)]

    def start(self, n_rounds: int = 100) -> None:
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
                if t % self.delta == 0:
                    shuffle(node_ids)

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
                        sample = choice(list(self.not_byzantine_indices),
                                        max(int(self.n_nodes * self.sampling_eval), 1))
                        ev = [self.nodes[i].evaluate()
                              for i in sample if self.nodes[i].has_test()]
                    else:
                        ev = [n.evaluate()
                              for _, n in self.nodes.items() if n.has_test()]
                    if ev:
                        self.notify_evaluation(t, True, ev)

                    if self.data_dispatcher.has_test():
                        if self.sampling_eval > 0:
                            ev = [self.nodes[i].evaluate(self.data_dispatcher.get_eval_set())
                                  for i in sample]
                        else:
                            ev = [n.evaluate(self.data_dispatcher.get_eval_set())
                                  for i, n in self.nodes.items() if i in self.not_byzantine_indices]
                        if ev:
                            self.notify_evaluation(t, False, ev)
                self.notify_timestep(t)

        except KeyboardInterrupt:
            LOG.warning("Simulation interrupted by user.")

        pbar.close()
        self.notify_end()
        return


class TokenizedGossipSimulatorByzantine(TokenizedGossipSimulator):
    def __init__(self,
                 nodes: Dict[int, GossipNode],
                 data_dispatcher: DataDispatcher,
                 token_account: TokenAccount,
                 utility_fun: Callable[[ModelHandler, ModelHandler, Message], int],
                 delta: int,
                 protocol: AntiEntropyProtocol,
                 # [0,1] - probability of a message being dropped
                 drop_prob: float = 0.,
                 # [0,1] - probability of a node to be online
                 online_prob: float = 1.,
                 delay: Delay = ConstantDelay(0),
                 # [0, 1] - percentage of nodes to evaluate
                 sampling_eval: float = 0.
                 ):
        super().__init__(nodes, data_dispatcher, token_account, utility_fun,
                         delta, protocol, drop_prob, online_prob, delay, sampling_eval)
        self.not_byzantine_indices = [i for i in range(self.n_nodes) if not isinstance(
            self.nodes[i].model_handler, AvoidReportMixin)]

    # docstr-coverage:inherited
    def start(self, n_rounds: int = 100) -> Tuple[List[float], List[float]]:
        """Starts the simulation.

        The simulation handles the messages exchange between the nodes for ``n_rounds`` rounds.
        If attached to a :class:`SimulationReport`, the report is updated at each time step,
        sent/fail message and evaluation.

        Parameters
        ----------
        n_rounds : int, default=100
            The number of rounds of the simulation.
        """
        node_ids = np.arange(self.n_nodes)
        pbar = track(range(n_rounds * self.delta), description="Simulating...")
        msg_queues = DefaultDict(list)
        rep_queues = DefaultDict(list)
        # avg_tokens = [0]
        try:
            for t in pbar:
                if t % self.delta == 0:
                    shuffle(node_ids)
                    # if t > 0:
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
                                                       sender_mh, msg)
                            reaction = self.accounts[msg.receiver].reactive(
                                utility)
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
                        sample = choice(list(self.not_byzantine_indices),
                                        max(int(self.n_nodes * self.sampling_eval), 1))
                        ev = [self.nodes[i].evaluate()
                              for i in sample if self.nodes[i].has_test()]
                    else:
                        ev = [n.evaluate()
                              for k, n in self.nodes.items() if n.has_test() if k in self.not_byzantine_indices]
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
