import random
import numpy as np
from numpy.random import randint, normal, rand
from numpy import ndarray
from torch import Tensor
from typing import Any, Optional, Union, Dict, Tuple
from . import CACHE, LOG
from .core import AntiEntropyProtocol, CreateModelMode, MessageType, Message
from .utils import choice_not_n
from .model.handler import ModelHandler, PartitionedTMH, SamplingTMH
from .model.sampling import TorchModelSampling

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["GossipNode",
           "PassThroughNode",
           "CacheNeighNode",
           "SamplingBasedNode",
           "PartitioningBasedNode",
           "PENSNode"]


class GossipNode():
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 known_nodes: Optional[np.ndarray]=None, #reachable nodes according to the network topology
                 sync: bool=True):
        r"""Class that represents a generic node in a gossip network. 

        A node is identified by its index and it is initialized with a fixed delay :math:`\Delta` that
        represents the idle time. The node can be either synchronous or asynchronous. In the former case,
        the node will time out exactly :math:`\Delta` time steps into the round. Thus it is assumed that 
        :math:`0 < \Delta <` `round_len`. In the latter case, the node will time out to every 
        :math:`\Delta` time steps. 

        Parameters
        ----------
        idx : int
            The node's index.
        data : tuple[Tensor, Optional[Tensor]] or tuple[ndarray, Optional[ndarray]]
            The node's data in the format :math:`(X_\text{train}, y_\text{train}), (X_\text{test}, y_\text{test})`
            where :math:`y_\text{train}` and :math:`y_\text{test}` can be `None` in the case of unsupervised learning.
            Similarly, :math:`X_\text{test}` and :math:`y_\text{test}` can be `None` in the case the node does not have
            a test set. 
        round_len : int
            The number of time units in a round.
        n_nodes : int
            The number of nodes in the network.
        model_handler : ModelHandler
            The object that handles the model learning/inference.
        known_nodes : np.ndarray, default=None
            The reachable nodes according to the network topology. If `None`, the node will be initialized with all.
        sync : bool, default=True
            Whether the node is synchronous with the round's length. In this case, the node will regularly time out 
            at the same point in the round. If `False`, the node will time out with a fixed delay. 
        """
        self.idx: int = idx
        self.data:  Union[Tuple[Tensor, Optional[Tensor]], Tuple[ndarray, Optional[ndarray]]] = data
        self.round_len: int = round_len
        self.n_nodes: int = n_nodes
        self.model_handler: ModelHandler = model_handler
        self.sync: bool = sync
        self.delay: int = randint(0, round_len) if sync else int(normal(round_len, round_len/10))
        self.known_nodes: Optional[np.ndarray] = list(np.where(known_nodes > 0)[-1]) if known_nodes is not None else None

    #def update_neighbors(self, neighbors: np.ndarray) -> None:
    #    self.known_nodes = list(np.where(neighbors > 0)[-1])

    def init_model(self, local_train: bool=True, *args, **kwargs) -> None:
        """Initializes the local model.

        Parameters
        ----------
        local_train : bool, default=True
            Whether the local model should be trained for with the local data after the initialization.
        """
        self.model_handler.init()
        if local_train:
            self.model_handler._update(self.data[0])

    def get_peer(self) -> int:
        """Picks a random peer from the reachable nodes.

        Returns
        -------
        int
            The index of the randomly selected peer.
        """
        if self.known_nodes is not None:
            return random.choice(self.known_nodes)
        return choice_not_n(0, self.n_nodes, self.idx)

    def timed_out(self, t: int) -> bool:
        """Checks whether the node has timed out.
        
        Parameters
        ----------
        t : int
            The current timestamp.
        
        Returns
        -------
        bool
            Whether the node has timed out.
        """
        return ((t % self.round_len) == self.delay) if self.sync else ((t % self.delay) == 0)

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Message:
        """Sends a message to the specified peer.

        The method actually prepares the message that will be sent to the peer.
        The sending is performed by the simluator and it may be delayed or it can fail.

        Parameters
        ----------
        t : int
            The current timestamp.
        peer : int
            The index of the peer node.
        protocol : AntiEntropyProtocol
            The protocol used to send the message.

        Returns
        -------
        Message
            The message to send.
        
        Raises
        ------
        ValueError
            If the protocol is not supported.
        
        See Also
        --------
        :class:`gossipy.simul.GossipSimulator`
        """
        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, peer, MessageType.PUSH, (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, peer, MessageType.PUSH_PULL, (key,))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        """Receives a message from the peer.

        After the message is received, the local model is updated and merged according to the `mode`
        of the model handler. In case of a pull/push-pull message, the local model is sent back to the
        peer.

        Parameters
        ----------
        t : int
            The current timestamp.
        msg : Message
            The received message.
        
        Returns
        -------
        Message or `None`
            The message to be sent back to the peer. If `None`, there is no message to be sent back.
        """
        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0])

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t, self.idx, msg.sender, MessageType.REPLY, (key,))
        return None

    def evaluate(self, ext_data: Optional[Any]=None) -> Dict[str, float]:
        """Evaluates the local model.

        Parameters
        ----------
        ext_data : Any, default=None
            The data to be used for evaluation. If `None`, the local test data will be used.
        
        Returns
        -------
        dict[str, float]
            The evaluation results. The keys are the names of the metrics and the values are
            the corresponding values.
        """
        if ext_data is None:
            return self.model_handler.evaluate(self.data[1])
        else:
            return self.model_handler.evaluate(ext_data)
    
    #CHECK: we need a more sensible check
    def has_test(self) -> bool:
        """Checks whether the node has a test set.

        Returns
        -------
        bool
            Whether the node has a test set.
        """
        if isinstance(self.data, tuple):
            return self.data[1] is not None
        else: return True
    

    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} #{self.idx} (Δ={self.delay})"


class PassThroughNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 known_nodes: Optional[np.ndarray]=None, #reachable nodes according to the network topology
                 sync=True):
        super(PassThroughNode, self).__init__(idx,
                                              data,
                                              round_len,
                                              n_nodes,
                                              model_handler,
                                              known_nodes,
                                              sync)
        self.n_neighs = len(self.known_nodes)

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer, 
                           MessageType.PUSH,
                           (key, self.n_neighs))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, self.n_neighs))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t:int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            
            (recv_model, deg) = msg.value
            recv_model = CACHE.pop(recv_model)
            if  rand() < min(1, deg / self.n_neighs):
                self.model_handler(recv_model, self.data[0])
            else: #PASSTHROUGH
                prev_mode = self.model_handler.mode
                self.model_handler.mode = CreateModelMode.PASS
                self.model_handler(recv_model, self.data[0])
                self.model_handler.mode = prev_mode

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, self.n_neighs))
        return None

# FIXME: something seems wrong with the implementation of this type of node
class CacheNeighNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 known_nodes: np.ndarray, #reachable nodes according to the network topology
                 sync: bool=True):
        super(CacheNeighNode, self).__init__(idx,
                                             data,
                                             round_len,
                                             n_nodes,
                                             model_handler,
                                             known_nodes,
                                             sync)
        key = self.model_handler.caching(self.idx)
        #for _ in range(len(self.known_nodes) - 1):
        #    self.model_handler._CACHE[key].add_ref()
        self.cache = {i : key for i in self.known_nodes}
                        
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            cached_model = CACHE.pop(self.cache[peer])
            self.model_handler(cached_model, self.data[0])
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            cached_model = CACHE.pop(self.cache[peer])
            self.model_handler(cached_model, self.data[0])
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key,))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            self.cache[sender] = recv_model

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            cached_model = CACHE.pop(self.cache[sender])
            self.model_handler(cached_model, self.data[0])
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key,))
        return None


class SamplingBasedNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: SamplingTMH, #object that handles the model learning/inference
                 known_nodes: np.ndarray, #reachable nodes according to the network topology
                 sync=True):
        super(SamplingBasedNode, self).__init__(idx,
                                                data,
                                                round_len,
                                                n_nodes,
                                                model_handler,
                                                known_nodes,
                                                sync)
                        
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, self.model_handler.sample_size))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, self.model_handler.sample_size))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, sample_size = msg.value
            recv_model = CACHE.pop(recv_model)
            sample = TorchModelSampling.sample(sample_size, recv_model.model)
            self.model_handler(recv_model, self.data[0], sample)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, self.model_handler.sample_size))
        return None


class PartitioningBasedNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: PartitionedTMH, #object that handles the model learning/inference
                 known_nodes: np.ndarray, #reachable nodes according to the network topology
                 sync=True):
        super(PartitioningBasedNode, self).__init__(idx,
                                                    data,
                                                    round_len,
                                                    n_nodes,
                                                    model_handler,
                                                    known_nodes,
                                                    sync)
                        
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, pid))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (key, pid))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type

        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, pid = msg.value
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0], pid)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = self.model_handler.caching(self.idx)
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, pid))
        return None



class PENSNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 known_nodes: np.ndarray, #reachable nodes according to the network topology
                 n_sampled: int=10, #value from the paper
                 m_top: int=2, #value from the paper
                 step1_rounds=200,
                 sync: bool=True):
        super(PENSNode, self).__init__(idx,
                                       data,
                                       round_len,
                                       n_nodes,
                                       model_handler,
                                       known_nodes,
                                       sync)
        assert self.model_handler.mode == CreateModelMode.MERGE_UPDATE, \
               "PENSNode can only be used with MERGE_UPDATE mode."
        self.cache = {}
        self.n_sampled = n_sampled
        self.m_top = m_top
        if self.known_nodes:
            self.neigh_counter = {i: 0 for i in self.known_nodes}
            self.selected = {i: 0 for i in self.known_nodes}
        else:
            self.neigh_counter = {i: 0 for i in range(self.n_nodes)}
            self.selected = {i: 0 for i in range(self.n_nodes)}
            del self.neigh_counter[self.idx] # remove itself from the dict
        self.step1_rounds = step1_rounds
        self.step = 1
        self.best_nodes = None
    
    def _select_neighbors(self) -> None:
        self.best_nodes = []
        for i, cnt in self.neigh_counter.items():
            if cnt > self.selected[i] * (self.m_top / self.n_sampled):
                self.best_nodes.append(i)
    
    def timed_out(self, t: int) -> int:
        if self.step == 1 and (t // self.round_len) >= self.step1_rounds:
            self.step = 2
            self._select_neighbors()
        peer = super().timed_out(t)
        if self.step == 1:
            self.selected[peer] += 1
        return peer
    
    def get_peer(self) -> int:
        if self.step == 1 or not self.best_nodes:
            return super().get_peer()
        return random.choice(self.best_nodes)

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol != AntiEntropyProtocol.PUSH:
            LOG.warning("PENSNode only supports PUSH protocol.")

        key = self.model_handler.caching(self.idx)
        return Message(t, self.idx, peer, MessageType.PUSH, (key,))
        

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value[0]
        if msg_type != MessageType.PUSH:
            LOG.warning("PENSNode only supports PUSH protocol.")

        if self.step == 1:
            evaluation = CACHE[recv_model].evaluate(self.data[0])
            # TODO: move performance metric as a parameter of the node
            self.cache[sender] = (recv_model, -evaluation["accuracy"]) # keep the last model for the peer 'sender'

            if len(self.cache) >= self.n_sampled:
                top_m = sorted(self.cache, key=lambda key: self.cache[key][1])[:self.m_top]
                recv_models = [CACHE.pop(self.cache[k][0]) for k in top_m]
                self.model_handler(recv_models, self.data[0])
                self.cache = {} # reset the cache
                for i in top_m:
                    self.neigh_counter[i] += 1
        else:
            recv_model = CACHE.pop(recv_model)
            self.model_handler(recv_model, self.data[0])
