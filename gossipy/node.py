from gossipy.model.sampling import TorchModelPartition, TorchModelSampling
import random
import numpy as np
from numpy.random import randint, normal, rand, binomial
from numpy import ndarray
from torch import Tensor
from typing import Any, Optional, Union, Dict, Tuple
from gossipy import CacheKey
from .utils import choice_not_n
from .model.handler import ModelHandler, PartitionedTMH, SamplingTMH
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

__all__ = ["GossipNode",
           "PassThroughNode",
           "CacheNeighNode",
           "SamplingBasedNode",
           "PartitioningBasedNode",
           "TokenAccount",
           "PurelyProactiveTokenAccount",
           "PurelyReactiveTokenAccount",
           "SimpleTokenAccount",
           "GeneralizedTokenAccount",
           "RandomizedTokenAccount"]


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
        self.known_nodes = list(np.where(known_nodes > 0)[-1]) if known_nodes is not None else None

    def init_model(self, *args, **kwargs) -> None:
        self.model_handler.init()

    def get_peer(self) -> int:
        if self.known_nodes is not None:
            return random.choice(self.known_nodes)
        return choice_not_n(0, self.n_nodes, self.idx)

    def timed_out(self, t: int) -> int:
        return ((t % self.round_len) == self.delay) if self.sync else ((t % self.delay) == 0)

    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:
        if protocol == AntiEntropyProtocol.PUSH:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t, self.idx, peer, MessageType.PUSH, (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t, self.idx, peer, MessageType.PUSH_PULL, (key,))
        else:
            raise ValueError("Unknown protocol %s." %protocol)

    def receive(self, t: int, msg: Message) -> Union[Message, None]:
        msg_type: MessageType
        recv_model: Any 
        msg_type, recv_model = msg.type, msg.value[0] if msg.value else None
        if msg_type == MessageType.PUSH or \
           msg_type == MessageType.REPLY or \
           msg_type == MessageType.PUSH_PULL:
            recv_model = self.model_handler.pop_cache(recv_model)
            self.model_handler(recv_model, self.data[0])

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t, self.idx, msg.sender, MessageType.REPLY, (key,))
        return None

    def evaluate(self, ext_data: Optional[Any]=None) -> Dict[str, float]:
        if ext_data is None:
            return self.model_handler.evaluate(self.data[1])
        else:
            return self.model_handler.evaluate(ext_data)
    
    #CHECK: we need a more sensible check
    def has_test(self) -> bool:
        if isinstance(self.data, tuple):
            return self.data[1] is not None
        else: return True


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
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t,
                           self.idx,
                           peer, 
                           MessageType.PUSH,
                           (key, self.n_neighs))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
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
            recv_model = self.model_handler.pop_cache(recv_model)
            if  rand() < min(1, deg / self.n_neighs):
                self.model_handler(recv_model, self.data[0])
            else: #PASSTHROUGH
                prev_mode = self.model_handler.mode
                self.model_handler.mode = CreateModelMode.PASS
                self.model_handler(recv_model, self.data[0])
                self.model_handler.mode = prev_mode

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, self.n_neighs))
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
                 sync: bool=True):
        super(CacheNeighNode, self).__init__(idx,
                                             data,
                                             round_len,
                                             n_nodes,
                                             model_handler,
                                             known_nodes,
                                             sync)
        key = CacheKey(self.idx, self.model_handler.n_updates)
        self.model_handler.push_cache(key, self.model_handler.copy())
        for _ in range(len(self.known_nodes) - 1):
            self.model_handler._CACHE[key].add_ref()
        self.cache = {i : key for i in self.known_nodes}
                        
    def send(self,
             t: int,
             peer: int,
             protocol: AntiEntropyProtocol) -> Union[Message, None]:

        if protocol == AntiEntropyProtocol.PUSH:
            cached_model = self.model_handler.pop_cache(self.cache[peer])
            self.model_handler(cached_model, self.data[0])
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key,))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            cached_model = self.model_handler.pop_cache(self.cache[peer])
            self.model_handler(cached_model, self.data[0])
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
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
            cached_model = self.model_handler.pop_cache(self.cache[sender])
            self.model_handler(cached_model, self.data[0])
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
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
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, self.model_handler.sample_size))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
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
            recv_model = self.model_handler.pop_cache(recv_model)
            sample = TorchModelSampling.sample(sample_size, recv_model.model)
            self.model_handler(recv_model, self.data[0], sample)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            key = CacheKey(self.idx, self.model_handler.n_updates)
            self.model_handler.push_cache(key, self.model_handler.copy())
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
            key = CacheKey(self.idx, str(self.model_handler.n_updates))
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t,
                           self.idx,
                           peer,
                           MessageType.PUSH,
                           (key, pid))
        elif protocol == AntiEntropyProtocol.PULL:
            return Message(t, self.idx, peer, MessageType.PULL, None)
        elif protocol == AntiEntropyProtocol.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = CacheKey(self.idx, str(self.model_handler.n_updates))
            self.model_handler.push_cache(key, self.model_handler.copy())
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
            recv_model = self.model_handler.pop_cache(recv_model)
            self.model_handler(recv_model, self.data[0], pid)

        if msg_type == MessageType.PULL or \
           msg_type == MessageType.PUSH_PULL:
            pid = np.random.randint(0, self.model_handler.tm_partition.n_parts)
            key = CacheKey(self.idx, str(self.model_handler.n_updates))
            self.model_handler.push_cache(key, self.model_handler.copy())
            return Message(t,
                           self.idx,
                           msg.sender,
                           MessageType.REPLY,
                           (key, pid))
        return None


class TokenAccount():
    def __init__(self):
        self.n_tokens = 0
    
    def add(self, n: int=1) -> None:
        self.n_tokens += n

    def sub(self, n: int=1) -> None:
        self.n_tokens = max(0, self.n_tokens - n)
    
    def proactive(self) -> float:
        raise NotImplementedError()

    def reactive(self, utility: int) -> int:
        raise NotImplementedError()


class PurelyProactiveTokenAccount(TokenAccount):
    def proactive(self) -> float:
        return 1

    def reactive(self, utility: int) -> int:
        return 0


class PurelyReactiveTokenAccount(TokenAccount):
    def __init__(self, k: int=1):
        super(PurelyReactiveTokenAccount, self).__init__()
        self.k = k

    def proactive(self) -> float:
        return 0

    def reactive(self, utility: int) -> int:
        return int(utility * self.k)


class SimpleTokenAccount(TokenAccount):
    def __init__(self, C: int=1):
        super(SimpleTokenAccount, self).__init__()
        assert C >= 1, "The capacity C must be strictly positive."
        self.capacity = C
    
    def proactive(self) -> float:
        return int(self.n_tokens >= self.capacity)

    def reactive(self, utility: int) -> int:
        return int(self.n_tokens > 0)


class GeneralizedTokenAccount(SimpleTokenAccount):
    def __init__(self, C: int, A: int): #1
        super(GeneralizedTokenAccount, self).__init__(C)
        assert C >= 1, "The capacity C must be positive."
        assert A >= 1, "The reactivity A must be positive."
        assert A <= C, "The capacity C must be greater or equal than the reactivity A."
        self.reactivity = A

    def reactive(self, utility: int) -> int:
        num = self.reactivity + self.n_tokens - 1
        return int(num / self.reactivity if utility > 0 else num / (2 * self.reactivity))


class RandomizedTokenAccount(GeneralizedTokenAccount):
    def __init__(self, C: int, A: int):
        super(RandomizedTokenAccount, self).__init__(C, A)
    
    def proactive(self) -> float:
        if self.n_tokens < self.reactivity - 1:
            return 0
        elif self.reactivity - 1 <= self.n_tokens <= self.capacity:
            return (self.n_tokens - self.reactivity + 1) / (self.capacity - self.reactivity + 1)
        else:
            return 1

    def reactive(self, utility: int) -> int:
        if utility > 0:
            r = self.n_tokens / self.reactivity
            return int(r) + binomial(1, r - int(r)) #randRound
        return 0
