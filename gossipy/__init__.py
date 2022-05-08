from typing import Any, Dict, Tuple
import logging
from rich.logging import RichHandler
from enum import Enum
import numpy as np
import torch
import random

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["LOG",
            "CACHE",
           #"node",
           #"simul",
           #"utils",
           #"data",
           #"model",
           #"flow_control",
           "set_seed",
           "DuplicateFilter",
           "CreateModelMode",
           "AntiEntropyProtocol",
           "MessageType",
           "Message",
           "CacheKey",
           "CacheItem",
           "Sizeable",
           "EqualityMixin",
           "Cache",
           "Delay",
           "UniformDelay",
           "LinearDelay"]


class DuplicateFilter(object):
    def __init__(self):
        """Removes duplicate log messages."""
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

logging.basicConfig(level=logging.INFO,
                    #format="[%(asctime)s]  %(message)s",
                    format="%(message)s",
                    datefmt='%d%m%y-%H:%M:%S',
                    handlers=[RichHandler()])


LOG = logging.getLogger("rich")
"""The logging handler that filters out duplicate messages."""

LOG.addFilter(DuplicateFilter())


def set_seed(seed=0) -> None:
    """Sets the seed for the random number generator.
    
    Parameters
    ----------
    seed : int, default=0
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CreateModelMode(Enum):
    """The mode for creating/updating the gossip model."""

    UPDATE = 1
    """Update the model with the local data."""
    MERGE_UPDATE = 2
    """Merge the models and then make an update."""
    UPDATE_MERGE = 3
    """Update the models with the local data and then merge the models."""
    PASS = 4
    """Do nothing."""


class AntiEntropyProtocol(Enum):
    """The overall protocol of the gossip algorithm."""

    PUSH = 1
    """Push the local model to the gossip node(s)."""
    PULL = 2
    """Pull the gossip model from the gossip node(s)."""
    PUSH_PULL = 3
    """Push the local model to the gossip node(s) and then pull the gossip model from the gossip node(s)."""


class MessageType(Enum):
    """The type of a message."""

    PUSH = 1
    """The message contains the model (and possibly additional information)"""
    PULL = 2
    """Asks for the model to the receiver."""
    REPLY = 3
    """The message is a response to a PULL message."""
    PUSH_PULL = 4
    """The message contains the model (and possibly additional information) and also asks for the model."""


class EqualityMixin(object):
    def __init__(self):
        """Mixin for equality comparison."""
        pass

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class Sizeable():
    def __init__(self):
        """The interface for objects that can be sized.
        
        Each class that implements this interface must define the method `get_size()`.
        """
        pass
    
    def get_size(self) -> int:
        """Returns the size of the object."""
        raise NotImplementedError()


class CacheKey(Sizeable):
    def __init__(self, *args):
        """The key for a cache item."""
        self.key: Tuple[Any, ...] = tuple(args)
    
    def get(self):
        """Returns the value of the cache item.

        Returns
        -------
        Any
            The value of the cache item.
        """
        return self.key
    
    def get_size(self) -> int:
        val = CACHE[self]
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
        """The class of an item in the cache.

        The constructor initializes the cache item with the specified value and with a single reference.

        Parameters
        ----------
        value : Any
            The value of the item.
        """
        self.value: Any = value
        self.refs: int = 1
    
    def add_ref(self) -> None:
        """Adds a reference to the item."""
        self.refs += 1
    
    def del_ref(self) -> Any:
        """Deletes a reference to the item.
        
        Returns
        -------
        Any
            The value of the unreferenced item.
        """
        self.refs -= 1
        return self.value
    
    def is_referenced(self) -> bool:
        """Returns True if the item is referenced, False otherwise.
        
        Returns
        -------
        bool
            `True` if the item is referenced, `False` otherwise.
        """
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
    
    def __repr__(self):
        return self.value.__repr__()


class Cache():
    _cache: Dict[CacheKey, CacheItem] = {}

    def push(self, key: CacheKey, value: Any):
        if key not in self._cache:
            self._cache[key] = CacheItem(value)
        else:
            self._cache[key].add_ref()
    
    def pop(self, key: CacheKey):
        if key not in self._cache:
            return None
        obj = self._cache[key].del_ref()
        if not self._cache[key].is_referenced():
            del self._cache[key]
        return obj
    
    def clear(self):
        self._cache.clear()
    
    def __getitem__(self, key: CacheKey):
        if key not in self._cache:
            return None
        return self._cache[key].value

    def load(self, cache_dict: Dict[CacheKey, Any]):
        self._cache = cache_dict
    
    def get_cache(self) -> Dict[CacheKey, Any]:
        return self._cache
    
    def __repr__(self):
        return str(self._cache)


CACHE = Cache()
"""The models cache. 

All models that are exchanged between nodes are temporarely stored in the cache.
If a model is needed by another node, it is retrieved from the cache and only one copy remains active in memory.
If a model is not referenced anymore, it is removed from the cache."""


class Message(Sizeable):
    def __init__(self,
                 timestamp: int,
                 sender: int,
                 receiver: int,
                 type: MessageType,
                 value: Tuple[Any, ...]):
        """A class representing a message.

        Parameters
        ----------
        timestamp : int
            The message's timestamp with the respect to the simulation time.
        sender : int
            The sender node id.
        receiver : int
            The receiver node id.
        type : MessageType
            The message type.
        value : tuple[Any, ...]
            The message's payload. The typical payload is a single item tuple containing the model (handler).
        """
        self.timestamp: int = timestamp
        self.sender: int = sender
        self.receiver: int = receiver
        self.type: MessageType = type
        self.value: Tuple[Any, ...] = value
    
    def get_size(self) -> int:
        """Computes and returns the estimated size of the message.

        The size is expressed in number of "atomic" values stored in the message.
        Atomic values are integers, floats, and booleans. Currently strings are not supported.

        Returns
        -------
        int
            The estimated size of the message.

        Raises
        ------
        TypeError
            If the message's payload contains values that are not atomic.
        """

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


class Delay():

    def __init__(self, delay: int=0):
        """A class representing a constant delay.

        Parameters
        ----------
        delay : int
            The constant delay in time units.
        """
        assert delay >= 0
        self.delay: int = delay
    
    def get(self, msg: Message) -> int:
        """Returns the delay for the specified message.

        The delay is fixed regardless of the specific message.

        Parameters
        ----------
        msg : Message
            The message for which the delay is computed.
        
        Returns
        -------
        int
            The delay in time units.
        """
        return self.delay


class UniformDelay(Delay):
    def __init__(self, min_delay: int, max_delay: int):
        """A class representing a uniform delay.
    
        Parameters
        ----------
        min_delay : int
            The minimum delay in time units.
        max_delay : int
            The maximum delay in time units.
        """
        assert min_delay <= max_delay and min_delay >= 0
        self.min_delay: int = min_delay
        self.max_delay: int = max_delay
    
    def get(self, msg: Message) -> int:
        """Returns the delay for the specified message.

        The delay is uniformly distributed between the minimum and maximum delay
        regardless of the specific message.

        Parameters
        ----------
        msg : Message
            The message for which the delay is computed.
        
        Returns
        -------
        int
            The delay in time units.
        """
        return np.random.randint(self.min_delay, self.max_delay+1)

class LinearDelay(Delay):
    def __init__(self, weight: float, constant: int):
        """A class representing a linear delay.

        Parameters
        ----------
        weight : float
            The weight of the delay.
        constant : int
            The constant delay in time units.
        """
        assert weight >= 0 and constant >= 0
        self.weight: float = weight
        self.constant: int = constant
    
    def get(self, msg: Message) -> int:
        """Returns the delay for the specified message.

        | The delay is linear with respect to the message's size and it is computed as follows:
        | delay = floor(weight * size(msg)) + constant.

        Parameters
        ----------
        msg : Message
            The message for which the delay is computed.
        
        Returns
        -------
        int
            The delay in time units.
        """
        return int(self.weight * msg.get_size()) + self.constant