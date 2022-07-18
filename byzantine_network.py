from typing import Union, List
import numpy as np

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = [
    "multi_clique_network",
    "ring_network"
]


def multi_clique_network(size: int, clique_size: Union[int, List[int]]):
    R = np.zeros((size, size))
    if isinstance(clique_size, int):
        sav = clique_size
        clique_size = [clique_size] * (size // clique_size)
        clique_size += [size % sav]

    assert sum(clique_size) == size, "clique_size is wrong. " + \
        str(clique_size)

    cumulated_clique_size = []
    total = 0
    for x in clique_size:
        cumulated_clique_size += [total]
        total += x

    print(cumulated_clique_size)

    connector_line = np.array(
        [1. if i in cumulated_clique_size else 0. for i in range(size)])

    for s, pos in zip(clique_size, cumulated_clique_size):
        R[pos, :] = connector_line
        R[:, pos] = connector_line
        R[pos:pos+s, pos:pos+s] = 1. - np.identity(s)
    return R


def ring_network(size: int):
    R = np.zeros((size, size))
    np.fill_diagonal(R[1:size, :size-1], 1.)
    np.fill_diagonal(R[0:size-1, 1:size], 1.)
    R[size-1, 0] = 1.
    R[0, size-1] = 1.
    return R
