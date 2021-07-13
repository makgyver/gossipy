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
        self.n_updates = 0
        self.known_nodes = np.where(known_nodes > 0)[0] if known_nodes is not None else None

    def init_model(self, *args, **kwargs) -> None:
        self.model_handler.init()

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
             protocol: AntiEntropyProtocol) -> Union[Message, None]:
        peer: int = self.get_peer()
        if protocol == AntiEntropyProtocol.PUSH:
            return Message(self.idx, peer, MessageType.PUSH, self.model_handler.copy())
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(self.idx, peer, MessageType.PULL, None)
        elif  protocol == AntiEntropyProtocol.PUSH_PULL:
            return Message(self.idx, peer, MessageType.PUSH_PULL, self.model_handler.copy())
        else:
            raise ValueError("Unknown protocol %s." %protocol)
        #return None

    def receive(self, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            self.model_handler(recv_model, self.data[0])
            self.n_updates += 1
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
    
    def has_test(self):
        return self.data[1] is not None


class UAGossipNode(GossipNode):
    def __init__(self,
                 idx: int,
                 data: Any,
                 round_len: int,
                 n_nodes: int,
                 model_handler: ModelHandler,
                 known_nodes: Optional[np.ndarray]=None,
                 sync=True):
        super(UAGossipNode, self).__init__(idx,
                                            data,
                                            round_len,
                                            n_nodes,
                                            model_handler,
                                            known_nodes,
                                            sync)
        if self.known_nodes is not None:
            self.peer_conf = np.zeros_like(self.known_nodes)
            self.peer_map = {n:i for i,n in enumerate(self.known_nodes)}
        else:
            self.peer_conf = np.zeros(self.n_nodes)
    
    def get_peer(self) -> int:
        if self.known_nodes is not None:
            return choice(self.known_nodes)#, size=1, p=self.peer_prob)[0]
        return choice_not_n(0, self.n_nodes, self.idx)

    def receive(self, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type = msg.type
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model, conf = msg.value
            id_peer: int = self.peer_map[msg.sender] if self.known_nodes is not None else msg.sender
            self.peer_conf[id_peer] = conf
            self.model_handler(recv_model, self.data[0])
            self.n_updates += 1
        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            conf: float = self._confidence()
            if (conf > np.mean(self.peer_conf)*.8):
                return Message(self.idx, msg.sender, MessageType.REPLY, (self.model_handler.copy(), conf))
        return None
    
    def _confidence(self) -> float:
        if not self.has_test(): return 1
        x, y = self.data[1]
        scores = self.model_handler.model(x)
        pred = torch.ones_like(scores)
        pred[scores <= .5] = 0
        #return accuracy_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten())
        mul = y.cpu().numpy().flatten() == pred.cpu().numpy().flatten()
        scores = scores.detach().cpu().numpy()
        scores[y == 0] = 1 - scores[y == 0]
        return np.dot(mul, scores).item()
    

class MABGossipNode(GossipNode):
    def __init__(self,
                 idx: int,
                 data: Any,
                 round_len: int,
                 n_nodes: int,
                 model_handler: ModelHandler,
                 known_nodes: Optional[np.ndarray]=None,
                 sync: bool=True):
        super(MABGossipNode, self).__init__(idx,
                                            data,
                                            round_len,
                                            n_nodes,
                                            model_handler,
                                            known_nodes,
                                            sync)
        n_bandits = len(self.known_nodes) if self.known_nodes is not None else n_nodes
        if self.known_nodes is not None:
            self.peer_map = {n:i for i, n in enumerate(self.known_nodes)}
        else:
            self.peer_map = {n:n for n in range(n_nodes)}
        self.mab = slots.MAB(n_bandits, live=True)

    def get_peer(self) -> int:
        if self.known_nodes is not None and len(self.known_nodes) == 1:
            return self.known_nodes[0]
        
        peer: int = self.mab.ucb()
        if self.known_nodes is not None:
            peer = self.known_nodes[peer]
        else:
            while peer == self.idx:
                peer = self.mab.ucb()
        return peer

    def receive(self, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            self.model_handler(recv_model, self.data[0])
            self.n_updates += 1

        if msg_type == MessageType.REPLY and self.n_nodes > 2:
            evaluation : float
            _, evaluation = recv_model.evaluate(self.data[1])
            peer = self.peer_map[msg.sender]
            self.mab.online_trial(bandit=peer, payout=evaluation)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            return Message(self.idx, msg.sender, MessageType.REPLY, self.model_handler.copy())
        return None