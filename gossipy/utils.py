import sys
import numpy as np
from numpy.random import randint

__all__ = ["print_flush", "choice_not_n", "sigmoid"]

def print_flush(text: str) -> None:
    print(text)
    sys.stdout.flush()

def choice_not_n(mn: int,
                 mx: int,
                 notn: int) -> int:
    c: int = randint(mn, mx)
    while c == notn:
        c = randint(mn, mx)
    return c

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))