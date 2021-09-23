import torch
from torch.nn.modules.loss import CrossEntropyLoss
from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode
from gossipy.model.handler import TorchModelHandler
from gossipy.model.nn import LogisticRegression
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler
from gossipy.simul import GossipSimulator, plot_evaluation


set_seed(98765)
X, y = load_classification_dataset("spambase", as_tensor=True)
data_handler = ClassificationDataHandler(X, y, test_size=.1)
dispatcher = DataDispatcher(data_handler, n=1000, eval_on_user=False)
topology = None

sim = GossipSimulator(
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    gossip_node_class=GossipNode,
    model_handler_class=TorchModelHandler,
    model_handler_params={
        "net" : LogisticRegression(data_handler.Xtr.shape[1], 2), #TorchMLP
        "optimizer" : torch.optim.SGD,
        "l2_reg": .001,
        "criterion" : CrossEntropyLoss(),
        "learning_rate" : .1,
        "create_model_mode" : CreateModelMode.MERGE_UPDATE},
    topology=topology,
    delay=(0, 10),
    sampling_eval=.1,
    round_synced=True
)
sim.init_nodes()
evaluation, evaluation_user = sim.start(n_rounds=1000)

plot_evaluation([evaluation])
plot_evaluation([evaluation_user])
