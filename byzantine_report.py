from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy.random import shuffle, random, choice
from typing import Callable, DefaultDict, Optional, Dict, List, Tuple
from rich.progress import track
import dill
import json
from typing import Callable, DefaultDict, Optional, Dict, List, Tuple


from gossipy import CACHE, LOG, CacheKey
from gossipy.core import AntiEntropyProtocol, Message, ConstantDelay, Delay
from gossipy.data import DataDispatcher
from gossipy.node import GossipNode
from gossipy.flow_control import TokenAccount
from gossipy.model.handler import ModelHandler
from gossipy.utils import StringEncoder
from gossipy.simul import SimulationReport

from byzantine_handler import AvoidReportMixin


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
    "ByzantineSimulationReport"
]


class ByzantineSimulationReport(SimulationReport):
    def __init__(self):
        """Same as SimulationReport class except it removes None values given by malicious nodes
        """
        super().__init__()

    def _collect_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        result_0 = None
        for r in results:
            if r != None:
                result_0 = r
                break
        if result_0 == None:
            return {}
        res = {k: [] for k in result_0}
        for k in res:
            for r in results:
                if r != None and r[k]:
                    res[k].append(r[k])
            res[k] = np.mean(res[k])
        return res
