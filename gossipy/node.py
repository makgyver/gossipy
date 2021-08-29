import numpy as np
from numpy.random import randint, normal, choice
import torch
import slots
from typing import Any, Optional, Union, Dict
from .utils import choice_not_n
from .model.handler import ModelHandler
from . import AntiEntropyProtocol, MessageType, Message

__all__ = ["GossipNode", "UAGossipNode", "MABGossipNode"]

class GossipNode():
    def __init__(self,
                 idx: int, #node's id
                 data: Any, #node's data
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
        #self.n_updates = 0
        self.known_nodes = np.where(known_nodes > 0)[0] if known_nodes is not None else None

    def init_model(self, *args, **kwargs) -> None:
        self.model_handler.init()

    #CHECK: peer sampling service
    def get_peer(self) -> int:
        if self.known_nodes is not None:
            return choice(self.known_nodes)
        return choice_not_n(0, self.n_nodes, self.idx)

    def timed_out(self, t: int) -> int:
        if self.sync:
            return t % self.round_len == self.delay
        else:
            return t % self.delay == 0

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:
        #peer: int = self.get_peer()
        if protocol == AntiEntropyProtocol.PUSH:
            return Message(self.idx, peer, MessageType.PUSH, self.model_handler.copy())
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            return Message(self.idx, peer, MessageType.PUSH_PULL, self.model_handler.copy())
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, msg: Message) -> Union[Message, None]:
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
            return Message(self.idx, msg.sender, MessageType.REPLY, self.model_handler.copy())
        return None

    def evaluate(self,
                 ext_data: Optional[Any]=None) -> Dict[str, float]:
        if ext_data is None:
            return self.model_handler.evaluate(self.data[1])
        else:
            return self.model_handler.evaluate(ext_data)
    
    def has_test(self) -> bool:
        return self.data[1] is not None
