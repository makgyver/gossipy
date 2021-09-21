import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
from gossipy import CacheItem, CacheKey, Message, Sizeable, set_seed, MessageType
from gossipy.model.handler import ModelHandler

def test_set_seed() -> None:
    set_seed(42)
    assert np.random.get_state()[1][0] == 42
    assert torch.initial_seed() == 42

def test_enums():
    pass

def test_Sizeable():
    s = Sizeable()
    with pytest.raises(NotImplementedError):
        s.get_size()

def test_Message():
    msg = Message(42, 1, 2, MessageType.PULL, 42)
    assert msg.timestamp == 42
    assert msg.sender == 1
    assert msg.receiver == 2
    assert msg.type == MessageType.PULL
    assert msg.value == 42
    assert msg.get_size() == 1
    assert str(msg) == "T42 [1 -> 2] {PULL}: 42"

    msg = Message(42, 1, 2, MessageType.PULL, None)
    assert msg.get_size() == 1
    assert str(msg) == "T42 [1 -> 2] {PULL}: ACK"

    msg = Message(42, 1, 2, MessageType.PULL, "test")
    with pytest.raises(TypeError):
        msg.get_size()
    
    class TempClass(Sizeable):
        def get_size(self) -> int:
            return 42

    msg = Message(42, 1, 2, MessageType.PULL, TempClass())
    assert msg.get_size() == 42

    msg = Message(42, 1, 2, MessageType.PULL, [42, TempClass(), None])
    assert msg.get_size() == 43

    msg = Message(42, 1, 2, MessageType.PULL, [42, TempClass(), "test"])
    with pytest.raises(TypeError):
        msg.get_size()

def test_CacheKey_CacheItem():

    k1 = CacheKey(1, 2)
    k2 = CacheKey("test")
    i1 = CacheItem(42)

    ModelHandler.push_cache(k1, i1)
    ModelHandler.push_cache(k2, "test")

    assert k1.get() == (1, 2)
    assert k1.get_size() == 1
    assert k2.get_size() == 0
    assert k1.__repr__() == str((1,2))
    s = set([k1])
    assert k1 in s
    k3 = CacheKey(1, 2)
    assert k1 == k3
    assert k1 != k2
    assert not (k1 == (1, 2))

    assert i1.value == 42
    assert i1.refs == 1
    i1.add_ref()
    assert i1.refs == 2
    r = i1.del_ref()
    assert i1.refs == 1
    assert r == 42
    assert i1.is_referenced()
    r = i1.del_ref()
    assert r == 42
    assert not i1.is_referenced()

    i3 = CacheItem([1,2,3, None, "test"])
    assert i3.get_size() == 3
    i4 = CacheItem("test")
    assert i4.get_size() == 0

    class TempClass(Sizeable):
        def get_size(self) -> int:
            return 17
    
    t = TempClass()
    i5 = CacheItem(t)
    assert i5.get_size() == 17
