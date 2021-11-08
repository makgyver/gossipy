from networkx import to_numpy_matrix
from networkx.generators.random_graphs import random_regular_graph
from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode
from gossipy.model.handler import MFModelHandler
from gossipy.data import RecSysDataDispatcher, load_recsys_dataset
from gossipy.data.handler import  RecSysDataHandler
from gossipy.simul import GossipSimulator, plot_evaluation


set_seed(98765)
ratings, nu, ni = load_recsys_dataset("ml-100k")
data_handler = RecSysDataHandler(ratings, nu, ni, .1, seed=42)
dispatcher = RecSysDataDispatcher(data_handler)
topology = to_numpy_matrix(random_regular_graph(20, nu, seed=42))

sim = GossipSimulator(
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    gossip_node_class=GossipNode,
    model_handler_class=MFModelHandler,
    model_handler_params={
        "dim" : 5,
        "n_items" : ni,
        "lam_reg": .1,
        "learning_rate" : .001,
        "create_model_mode" : CreateModelMode.MERGE_UPDATE},
    topology=topology,
    delay=(0, 10),
    #sampling_eval=.1,
    round_synced=True
)
sim.init_nodes()
evaluation, evaluation_user = sim.start(n_rounds=500)

plot_evaluation([evaluation])
plot_evaluation([evaluation_user])
