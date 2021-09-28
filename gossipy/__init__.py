from typing import Any, Tuple
import logging
from enum import Enum
import numpy as np
import torch

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["node",
           "simul",
           "utils",
           "data",
           "model",
           "set_seed",
           "DuplicateFilter",
           "CreateModelMode",
           "AntiEntropyProtocol",
           "MessageType",
           "CacheKey",
           "CacheItem"]


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]  %(message)s",
                    datefmt='%d%m%y-%H:%M:%S')

LOG = logging.getLogger("gossipy")
LOG.addFilter(DuplicateFilter())


def set_seed(seed=0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class CreateModelMode(Enum):
    UPDATE = 1
    MERGE_UPDATE = 2
    UPDATE_MERGE = 3
    PASS = 4


class AntiEntropyProtocol(Enum):
    PUSH = 1,
    PULL = 2,
    PUSH_PULL = 3


class MessageType(Enum):
    PUSH = 1,
    PULL = 2,
    REPLY = 3,
    PUSH_PULL = 4


class EqualityMixin(object):
    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class Sizeable():
    def get_size(self) -> int:
        raise NotImplementedError()


class CacheKey(Sizeable):
    def __init__(self, *args):
        self.key = tuple(args)
    
    def get(self):
        return self.key
    
    def get_size(self) -> int:
        from gossipy.model.handler import ModelHandler
        val = ModelHandler._CACHE[self].value
        if isinstance(val, (float, int, bool)): return 1
        elif isinstance(val, Sizeable): return val.get_size()
        else: 
            LOG.warning("Impossible to compute the size of %s. Set to 0." %val)
            return 0
    
    def __repr__(self):
        return str(self.key)
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CacheKey):
            return self.key == other.key
        return False

    def __ne__(self, other: Any):
        return not (self == other)


class CacheItem(Sizeable):
    def __init__(self, value: Any):
        self.value = value
        self.refs = 1
    
    def add_ref(self):
        self.refs += 1
    
    def del_ref(self):
        self.refs -= 1
        return self.value
    
    def is_referenced(self):
        return self.refs > 0
    
    def get_size(self) -> int:
        if isinstance(self.value, (tuple, list)):
            sz: int = 0
            for t in self.value:
                if t is None: continue
                if isinstance(t, (float, int, bool)): sz += 1
                elif isinstance(t, Sizeable): sz += t.get_size()
                else: 
                    LOG.warning("Impossible to compute the size of %s. Set to 0." %t)
            return max(sz, 1)
        elif isinstance(self.value, Sizeable):
            return self.value.get_size()
        elif isinstance(self.value, (float, int, bool)):
            return 1
        else:
            LOG.warning("Impossible to compute the size of %s. Set to 0." %self.value)
            return 0


class Message(Sizeable):
    def __init__(self,
                 timestamp: int,
                 sender: int,
                 receiver: int,
                 type: MessageType,
                 value: Tuple[Any, ...]):
        self.timestamp = timestamp
        self.sender = sender
        self.receiver = receiver
        self.type = type
        self.value = value
    
    def get_size(self) -> int:
        if self.value is None: return 1
        if isinstance(self.value, (tuple, list)):
            sz: int = 0
            for t in self.value:
                if t is None: continue
                if isinstance(t, (float, int, bool)): sz += 1
                elif isinstance(t, Sizeable): sz += t.get_size()
                else: raise TypeError("Cannot compute the size of the payload!")
            return max(sz, 1)
        elif isinstance(self.value, Sizeable):
            return self.value.get_size()
        elif isinstance(self.value, (float, int, bool)):
            return 1
        else:
            raise TypeError("Cannot compute the size of the payload!")
        
    def __repr__(self) -> str:
        s: str = "T%d [%d -> %d] {%s}: " %(self.timestamp,
                                           self.sender,
                                           self.receiver,
                                           self.type.name)
        s += "ACK" if self.value is None else str(self.value)
        return s
