from networkx import to_numpy_matrix
from networkx.generators.random_graphs import random_regular_graph
from gossipy import UniformDelay, set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode
from gossipy.model.handler import MFModelHandler
from gossipy.data import RecSysDataDispatcher, load_recsys_dataset
from gossipy.data.handler import  RecSysDataHandler
from gossipy.simul import GossipSimulator, repeat_simulation


set_seed(98765)
ratings, nu, ni = load_recsys_dataset("ml-100k")
data_handler = RecSysDataHandler(ratings, nu, ni, .1, seed=42)
dispatcher = RecSysDataDispatcher(data_handler)
topology = to_numpy_matrix(random_regular_graph(20, nu, seed=42))

simulator = GossipSimulator(
    data_dispatcher=dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH, 
    gossip_node_class=GossipNode,
    gossip_node_params={},
    model_handler_class=MFModelHandler,
    model_handler_params={
        "dim" : 5,
        "n_items" : ni,
        "lam_reg": .1,
        "learning_rate" : .001,
        "create_model_mode" : CreateModelMode.MERGE_UPDATE},
    topology=topology,
    delay=UniformDelay(0, 10),
    #sampling_eval=.1,
    round_synced=True
)

res = repeat_simulation(
    gossip_simulator=simulator,
    n_rounds=100, #500
    repetitions=1,
    verbose=True
)