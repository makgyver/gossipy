from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
from scipy.sparse import csr_matrix

from . import Sizeable

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["CreateModelMode",
           "AntiEntropyProtocol",
           "MessageType",
           "Message",
           "Delay",
           "UniformDelay",
           "LinearDelay",
           "P2PNetwork",
           "StaticP2PNetwork"]


class CreateModelMode(Enum):
    """The mode for creating/updating the gossip model."""

    UPDATE = 1
    """Update the model with the local data."""

    MERGE_UPDATE = 2
    """Merge the models and then make an update using the local data."""

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
    """Push the local model to the gossip node(s) and then pull the gossip model from the gossip \
        node(s)."""


class MessageType(Enum):
    """The type of a message."""

    PUSH = 1
    """The message contains the model (and possibly additional information)"""

    PULL = 2
    """Asks for the model to the receiver."""

    REPLY = 3
    """The message is a response to a PULL message."""

    PUSH_PULL = 4
    """The message contains the model (and possibly additional information) and also asks for the \
        model to the receiver."""


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
            The message's timestamp with the respect to the simulation time. The timestamp refers
            to the moment when the message is sent.
        sender : int
            The sender node id.
        receiver : int
            The receiver node id.
        type : MessageType
            The message type.
        value : tuple[Any, ...] or None
            The message's payload. The typical payload is a single item tuple containing the model
            (handler). If the value is None, the message represents an ACK.
        """

        self.timestamp: int = timestamp
        self.sender: int = sender
        self.receiver: int = receiver
        self.type: MessageType = type
        self.value: Tuple[Any, ...] = value
    
    def get_size(self) -> int:
        """Computes and returns the estimated size of the message.

        The size is expressed in number of "atomic" values stored in the message.
        Atomic values are integers, floats, and booleans. 
        
        Note
        ----
        Currently strings are not supported.

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


class Delay(ABC):
    """A class representing a delay.

    The delay is a function of a message and returns the delay in simulation time units.
    """

    @abstractmethod
    def get(self, msg: Message) -> int:
        """Returns the delay for the specified message.

        Parameters
        ----------
        msg : Message
            The message for which the delay is computed.
        
        Returns
        -------
        int
            The delay in time units.
        """

        pass


class ConstantDelay(Delay):
    _delay: int

    def __init__(self, delay: int=0):
        """A class representing a constant delay.

        Parameters
        ----------
        delay : int
            The constant delay in time units.
        """

        assert delay >= 0, "Delay must be non-negative!"
        self._delay = delay
    
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

        return self._delay
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return "ConstantDelay(%d)" %self._delay


class UniformDelay(Delay):
    _min_delay: int
    _max_delay: int

    def __init__(self, min_delay: int, max_delay: int):
        """A class representing a uniform delay.
    
        Parameters
        ----------
        min_delay : int
            The minimum delay in time units.
        max_delay : int
            The maximum delay in time units.
        """

        assert min_delay <= max_delay and min_delay >= 0, \
            "The minimum delay must be non-negative and less than or equal to the maximum delay!"
        self._min_delay = min_delay
        self._max_delay = max_delay
    
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

        return np.random.randint(self._min_delay, self._max_delay+1)
    
    def __str__(self) -> str:
        return "UniformDelay(%d, %d)" %(self._min_delay, self._max_delay) 


class LinearDelay(Delay):
    _overhead: int
    _timexunit: float
    
    def __init__(self, timexunit: float, overhead: int):
        """A class representing a linear delay.

        | The linear delay is computed as a fixed overhead plus a quantity proportional to 
        | the message's size. :class:`LinearDelay` can mimic the behavior of  
        | :class:`ConstantDelay`, i.e., LinearDelay(0, x) is equivalent to ConstantDelay(x).

        Parameters
        ----------
        timexunit : float
            The time unit delay per size unit.
        overhead : int
            The overhead delay (in time units) to apply to each message.
        """

        assert timexunit >= 0 and overhead >= 0
        self._timexunit = timexunit
        self._overhead = overhead
    
    def get(self, msg: Message) -> int:
        """Returns the delay for the specified message.

        | The delay is linear with respect to the message's size and it is computed as follows:
        | `delay = floor(timexunit * size(msg)) + overhead`.
        | This type of delay allows to simulate the transmission time which is a linear function
        | of the size of the message.

        Parameters
        ----------
        msg : Message
            The message for which the delay is computed.
        
        Returns
        -------
        int
            The delay in time units.
        """

        return int(self._timexunit * msg.get_size()) + self._overhead
    
    def __str__(self) -> str:
        return "LinearDelay(time_x_unit=%d, overhead=%d)" %(self._timexunit, self._overhead) 



class P2PNetwork(ABC):
    _topology: Union[None, csr_matrix, np.ndarray]
    _num_nodes: int

    def __init__(self, num_nodes: int, topology: Optional[Union[np.ndarray, csr_matrix]]=None):
        """Abstract class representing a network topology.

        Parameters
        ----------
        num_nodes : int
            The number of nodes in the network.
        topology : Optional[Union[np.ndarray, csr_matrix]], default=None
            The adjacency matrix of the network topology. If None, the network is considered
            to be a fully connected network.
        """

        if topology is None: assert num_nodes > 0, "The number of nodes must be positive!"
        else: num_nodes == topology.shape[0], \
            "The number of nodes must match the number of rows of the topology!"
        
        self._num_nodes = num_nodes
        self._topology = {}

        if topology is not None:
            if isinstance(topology, np.ndarray):
                for node in range(num_nodes):
                    self._topology[node] = list(np.where(topology[node, :] > 0)[-1])
            elif isinstance(topology, csr_matrix):
                for node in range(num_nodes):
                    self._topology[node] = list(topology.getrow(node).nonzero()[-1])
        else:
            self._topology = {i: None for i in range(num_nodes)}

    def size(self, node: Optional[int]=None) -> int:
        if node:
            return len(self._topology[node]) if self._topology[node] else self._num_nodes - 1
        return self._num_nodes

    @abstractmethod
    def get_peers(self, node_id: int):
        pass


class StaticP2PNetwork(P2PNetwork):
    def get_peers(self, node_id: int) -> List[int]:
        assert 0 <= node_id < self._num_nodes
        return self._topology[node_id]