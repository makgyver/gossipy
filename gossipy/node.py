import random
import numpy as np
from numpy.random import randint, normal, rand
from numpy import ndarray
from torch import Tensor
from typing import Any, Optional, Union, Dict, Tuple
from .utils import choice_not_n
from .model.handler import ModelHandler
from . import AntiEntropyProtocol, CreateModelMode, MessageType, Message

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["GossipNode", "PassThroughNode", "CacheNeighNode"]

class GossipNode():
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 known_nodes: Optional[np.ndarray]=None, #reachable nodes according to the network topology
                 sync=True):
        self.idx = idx
        self.data = data
        self.round_len = round_len
        self.n_nodes = n_nodes
        self.model_handler = model_handler
        self.sync = sync
        self.delay = randint(0, round_len) if sync else int(normal(round_len, round_len/10))
        self.known_nodes = list(np.where(known_nodes > 0)[1]) if known_nodes is not None else None

    def init_model(self, *args, **kwargs) -> None:
        self.model_handler.init()

    def get_peer(self) -> int:
        if self.known_nodes is not None:
            return random.choice(self.known_nodes)
        return choice_not_n(0, self.n_nodes, self.idx)

    def timed_out(self, t: int) -> int:
        if self.sync:
            return (t % self.round_len) == self.delay
        else:
            return (t % self.delay) == 0

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:
        if protocol == AntiEntropyProtocol.PUSH:
            return Message(t, self.idx, peer, MessageType.PUSH, self.model_handler.copy())
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            return Message(t, self.idx, peer, MessageType.PUSH_PULL, self.model_handler.copy())
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            self.model_handler(recv_model, self.data[0])
            #self.n_updates += 1
        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            return Message(t, self.idx, msg.sender, MessageType.REPLY, self.model_handler.copy())
        return None

    def evaluate(self,
                 ext_data: Optional[Any]=None) -> Dict[str, float]:
        if ext_data is None:
            return self.model_handler.evaluate(self.data[1])
        else:
            return self.model_handler.evaluate(ext_data)
    
    def has_test(self) -> bool:
        return self.data[1] is not None


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
            return Message(t,
                           self.idx,
                           peer, 
                           MessageType.PUSH,
                           (self.model_handler.copy(), self.n_neighs))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           (self.model_handler.copy(), self.n_neighs))
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
            if  rand() < min(1, deg / self.n_neighs):
                self.model_handler(recv_model, self.data[0])
            else: #PASSTHROUGH
                prev_mode = self.model_handler.mode
                self.model_handler.mode = CreateModelMode.PASS
                self.model_handler(recv_model, self.data[0])
                self.model_handler.mode = prev_mode

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (self.model_handler.copy(), self.n_neighs))
        return None


class CacheNeighNode(GossipNode):
    def __init__(self,
                 idx: int, #node's id
                 data: Union[Tuple[Tensor, Optional[Tensor]],
                             Tuple[ndarray, Optional[ndarray]]], #node's data
                 round_len: int, #round length
                 n_nodes: int, #number of nodes in the network
                 model_handler: ModelHandler, #object that handles the model learning/inference
                 known_nodes: np.ndarray, #reachable nodes according to the network topology
                 sync=True):
        super(CacheNeighNode, self).__init__(idx,
                                             data,
                                             round_len,
                                             n_nodes,
                                             model_handler,
                                             known_nodes,
                                             sync)
        self.cache = {i : self.model_handler.copy() for i in self.known_nodes}
                        
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            self.model_handler(self.cache[peer], self.data[0])
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           self.model_handler.copy())
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            self.model_handler(self.cache[peer], self.data[0])
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH_PULL,
                           self.model_handler.copy())
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        sender, msg_type, recv_model = msg.sender, msg.type, msg.value
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            self.cache[sender] = recv_model.copy()

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            self.model_handler(self.cache[sender], self.data[0])
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           self.model_handler.copy())
        return None