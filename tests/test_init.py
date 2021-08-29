import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath('..'))
from gossipy import Message, Sizeable, set_seed, MessageType

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
    msg = Message(1, 2, MessageType.PULL, 42)
    assert msg.sender == 1
    assert msg.receiver == 2
    assert msg.type == MessageType.PULL
    assert msg.value == 42
    assert msg.get_size() == 1
    assert str(msg) == "[1 -> 2]{PULL}: 42"

    msg = Message(1, 2, MessageType.PULL, None)
    assert msg.get_size() == 1
    assert str(msg) == "[1 -> 2]{PULL}: ACK"

    msg = Message(1, 2, MessageType.PULL, "test")
    with pytest.raises(TypeError):
        msg.get_size()
    
    class TempClass(Sizeable):
        def get_size(self) -> int:
            return 42

    msg = Message(1, 2, MessageType.PULL, TempClass())
    assert msg.get_size() == 42

    msg = Message(1, 2, MessageType.PULL, [42, TempClass(), None])
    assert msg.get_size() == 43

    msg = Message(1, 2, MessageType.PULL, [42, TempClass(), "test"])
    with pytest.raises(TypeError):
        msg.get_size()
    