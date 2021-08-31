from typing import Any
import logging
from enum import Enum
import numpy as np
import torch

__all__ = ["node", "simul", "utils", "data", "model", "set_seed", "CreateModelMode", "AntiEntropyProtocol", "MessageType"]

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]  %(message)s",
                    datefmt='%H:%M:%S-%d%m%y')

LOG = logging.getLogger(__name__)


def set_seed(seed=0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class CreateModelMode(Enum):
    UPDATE = 1
    MERGE_UPDATE = 2
    UPDATE_MERGE = 3


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


class Message(Sizeable):
    def __init__(self,
                 sender: int,
                 receiver: int,
                 type: MessageType,
                 value: Any):
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
        
    def __str__(self) -> str:
        s: str = "[%d -> %d]{%s}: " %(self.sender, self.receiver, self.type.name)
        s += "ACK" if self.value is None else str(self.value)
        return s