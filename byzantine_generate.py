from __future__ import annotations
from typing import Tuple
import torch
from gossipy.data import DataDispatcher
from gossipy.model.handler import ModelHandler
from gossipy.node import GossipNode
from gossipy.core import P2PNetwork
from random import shuffle

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


def generate_nodes(cls,
                   data_dispatcher: DataDispatcher,
                   p2p_net: P2PNetwork,
                   model_proto: Union[ModelHandler, List[Union[Tuple[int, ModelHandler], Tuple[int, ModelHandler, bool]]]],
                   round_len: int,
                   sync: bool,
                   to_shuffle: bool = False,
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
    to_shuffle : bool
        Wether different types of nodes should be shuffled or not

    Returns
    -------
    Dict[int, GossipNode]
        The generated nodes.
    """

    if (isinstance(model_proto, ModelHandler)):
        return GossipNode.generate(cls, data_dispatcher, p2p_net, model_proto, round_len, sync, *kwargs)

    indices = list(range(p2p_net.size()))
    if to_shuffle:
        shuffle(indices)

    nodes = {}
    idx = 0
    idx_dispacher = 0
    for nb_type in model_proto:
        for j in range(nb_type[0]):
            if (len(nb_type) > 2 and not nb_type[2]):
                nodes[indices[idx]] = cls(idx=idx,
                                          # Keep this form, for check of test datast to work
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
