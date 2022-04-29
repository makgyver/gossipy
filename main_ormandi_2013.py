from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode
from gossipy.model.handler import PegasosHandler
from gossipy.model.nn import Pegasos
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, repeat_simulation

# AUTHORSHIP
__version__ = "0.0.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2021, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

set_seed(42)
X, y = load_classification_dataset("spambase", as_tensor=True)
y = 2*y - 1 #convert 0/1 labels to -1/1
data_handler = ClassificationDataHandler(X, y, test_size=.1)

simulator = GossipSimulator(
    data_dispatcher=DataDispatcher(data_handler, eval_on_user=False),
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    gossip_node_class=GossipNode,
    gossip_node_params={},
    model_handler_class=PegasosHandler,
    model_handler_params={
        "net" : Pegasos(data_handler.size(1)),
        "lam" : .01,
        "create_model_mode" : CreateModelMode.MERGE_UPDATE
    },
    topology=None,
    sampling_eval=.1,
    round_synced=False
)

res = repeat_simulation(
    gossip_simulator=simulator,
    n_rounds=100,
    repetitions=1,
    verbose=True
)


