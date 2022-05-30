from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import logging
from rich.logging import RichHandler
import numpy as np
import torch
import random

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
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
           #"DuplicateFilter",
           "CacheKey",
           "CacheItem",
           "Sizeable",
           #"EqualityMixin",
           "Cache"]


# Undocumented class
class DuplicateFilter(object):
    # docstr-coverage:excused `internal class to handle logging`
    def __init__(self):
        self.msgs = set()

    # docstr-coverage:excused `internal class to handle logging`
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

    The seed is set for numpy, torch and random.
    
    Parameters
    ----------
    seed : int, default=0
        The seed to set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Sizeable(ABC):
    def __init__(self):
        """The interface for objects that can be sized.
        
        Each class that implements this interface must define the method :func:`get_size`.
        """

        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Returns the size of the object.

        The size is intended to be the number of "atomic" objects that the object contains.
        For example, a list of integers would have a size of the number of integers.
        
        Returns
        -------
        int
            The size of the object.
        """

        pass


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
    
    # docstr-coverage:inherited
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
    _value: Any
    _refs: int

    def __init__(self, value: Any):
        """The class of an item in the cache.

        The constructor initializes the cache item with the specified value and with a single reference.

        Parameters
        ----------
        value : Any
            The value of the item.
        """

        self._value = value
        self._refs = 1
    
    def add_ref(self) -> None:
        """Adds a reference to the item."""

        self._refs += 1
    
    def del_ref(self) -> Any:
        """Deletes a reference to the item.
        
        Returns
        -------
        Any
            The value of the unreferenced item.
        """

        self._refs -= 1
        return self._value
    
    def is_referenced(self) -> bool:
        """Returns True if the item is referenced, False otherwise.
        
        Returns
        -------
        bool
            `True` if the item is referenced at least once, `False` otherwise.
        """

        return self._refs > 0
    
    # docstr-coverage:inherited
    def get_size(self) -> int:
        if isinstance(self._value, (tuple, list)):
            sz: int = 0
            for t in self._value:
                if t is None: continue
                if isinstance(t, (float, int, bool)): sz += 1
                elif isinstance(t, Sizeable): sz += t.get_size()
                else: 
                    LOG.warning("Impossible to compute the size of %s. Set to 0." %t)
            return max(sz, 1)
        elif isinstance(self._value, Sizeable):
            return self._value.get_size()
        elif isinstance(self._value, (float, int, bool)):
            return 1
        else:
            LOG.warning("Impossible to compute the size of %s. Set to 0." %self._value)
            return 0
    
    def get(self) -> Any:
        """Returns the value.

        Returns
        -------
        Any
            The value of the item.
        """

        return self._value

    def __repr__(self):
        return self._value.__repr__()
    
    def __str__(self) -> str:
        return f"CacheItem({str(self._value)})"


class Cache():
    _cache: Dict[CacheKey, CacheItem] = {}

    def __init__(self):
        """This class represent a cache.
        
        Items are stored in the cache to keep in memory only a single copy of each item.
        A cached item (wrapped in :class:`CacheItem`) is kept in the cache until it is not 
        referenced anymore. In such a case, it is automatically deleted from the cache.
        To each item is associated a unique key of type :class:`CacheKey`.
        """

        pass

    def push(self, key: CacheKey, value: Any):
        """Pushes an item into the cache.

        Parameters
        ----------
        key : CacheKey
            The key associated to the item.
        value : Any
            The value of the item. The value will be wrapped into a :class:`CacheItem` object before
            being stored in the cache.
        """

        if key not in self._cache:
            self._cache[key] = CacheItem(value)
        else:
            self._cache[key].add_ref()
    
    def pop(self, key: CacheKey):
        """Retrieve an item from the cache.

        If the item to retrieve is not in the cache, i.e., the key is not valid, None is returned.
        Otherwise, the item is returned and a reference to the item is deleted from the cache.
        If the item is not referenced anymore, it is automatically deleted from the cache.

        Parameters
        ----------
        key : CacheKey
            The key associated to the item to retrieve.

        Returns
        -------
        Any
            The value of the item.
        """

        if key not in self._cache:
            return None
        obj = self._cache[key].del_ref()
        if not self._cache[key].is_referenced():
            del self._cache[key]
        return obj
    
    def clear(self):
        """Clears the cache."""

        self._cache.clear()
    
    def __getitem__(self, key: CacheKey):
        if key not in self._cache:
            return None
        return self._cache[key].get()

    def load(self, cache_dict: Dict[CacheKey, Any]):
        """Loads the cache from a dictionary.

        Parameters
        ----------
        cache_dict : dict[CacheKey, Any]
            The dictionary containing the cache.
        """

        self._cache = cache_dict
    
    def get_cache(self) -> Dict[CacheKey, Any]:
        """Returns the cache.

        Returns
        -------
        dict[CacheKey, Any]
            The cache.
        """

        return self._cache
    
    def __repr__(self):
        return str(self)
    
    def __str__(self) -> str:
        return str(self._cache)


CACHE = Cache()
"""The models' cache. 

All models that are exchanged between nodes are temporarely stored in the cache.
If a model is needed by another node, it is retrieved from the cache and only one copy remains active in memory.
If a model is not referenced anymore, it is automatically removed from the cache.
The models contained in the cache are a deep copy of the models stored in the nodes.
"""
