from __future__ import annotations
from typing import Tuple
import torch
from gossipy.data import DataDispatcher
from gossipy.model.handler import ModelHandler
from gossipy.node import GossipNode
from gossipy.core import P2PNetwork
from random import shuffle
from enum import Enum

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
    "GossipSimulatorByzantine"
]


class GenerationType(Enum):
    NORMAL = 0
    SHUFFLED = 1
    LOW_DEGREE = 2
    HIGH_DEGREE = 3


def generate_nodes(cls,
                   data_dispatcher: DataDispatcher,
                   p2p_net: P2PNetwork,
                   model_proto: Union[ModelHandler, List[Union[Tuple[int, ModelHandler], Tuple[int, ModelHandler, bool]]]],
                   round_len: int,
                   sync: bool,
                   generation_type: GenerationType = GenerationType.NORMAL,
                   **kwargs) -> Dict[int, GossipNode]:
    """Generates a set of nodes.

    Parameters
    ----------
    data_dispatcher : DataDispatcher
        The data dispatcher used to distribute the data among the nodes.
    p2p_net : P2PNetwork
        The peer-to-peer network topology.
    model_proto : ModelHandler
        The model handler prototype or a list of tuples (number, ModelHandler) or a list of tuples (number, ModelHandler, data_given)
    round_len : int
        The length of a round in time units.
    sync : bool
        Whether the nodes are synchronized with the round length or not.
    generation_type : GenerationType
        What order should be taken to generate nodes

    Returns
    -------
    Dict[int, GossipNode]
        The generated nodes.
    """

    if (isinstance(model_proto, ModelHandler)):
        return GossipNode.generate(cls, data_dispatcher, p2p_net, model_proto, round_len, sync, *kwargs)

    if (generation_type == GenerationType.HIGH_DEGREE or generation_type == GenerationType.LOW_DEGREE):
        indices = [x[1] for x in sorted([(len(p2p_net.get_peers(i)), i)
                                         for i in range(p2p_net.size())], reverse=(generation_type == GenerationType.HIGH_DEGREE))]
    else:
        indices = list(range(p2p_net.size()))
        if generation_type == GenerationType.SHUFFLED:
            shuffle(indices)

    nodes = {}
    idx = 0
    idx_dispacher = 0
    for nb_type in model_proto:
        for j in range(nb_type[0]):
            if (len(nb_type) > 2 and not nb_type[2]):
                nodes[indices[idx]] = cls(idx=idx,
                                          # Keep this form, for dataset testset check to work
                                          data=(None, None),
                                          round_len=round_len,
                                          model_handler=nb_type[1].copy(),
                                          p2p_net=p2p_net,
                                          sync=sync,
                                          **kwargs)
            else:
                nodes[indices[idx]] = cls(idx=idx,
                                          data=data_dispatcher[idx_dispacher],
                                          round_len=round_len,
                                          model_handler=nb_type[1].copy(),
                                          p2p_net=p2p_net,
                                          sync=sync,
                                          **kwargs)
                idx_dispacher += 1
            idx += 1
    assert(idx == p2p_net.size())
    return nodes
