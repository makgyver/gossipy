import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

from gossipy import set_seed, AntiEntropyProtocol, CreateModelMode
from gossipy.node import GossipNode
from gossipy.model.handler import KMeansHandler
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.data.handler import ClusteringDataHandler
from gossipy.simul import GossipSimulator, plot_evaluation, repeat_simulation

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
X, y = load_classification_dataset("spambase", normalize=True, as_tensor=True)
data_handler = ClusteringDataHandler(X, y)

# Baseline - the sklearn implementation is highly unstable in terms of NMI
def kmeans(X, k=2, max_iterations=400):
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
    for i in range(max_iterations):
        centroids = np.vstack([X[P == i, :].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
        if np.array_equal(P,tmp):
            break
        P = tmp
    return P

P = kmeans(X.numpy(), k=2)
print("K-means NMI:", nmi(y.numpy(), P))

# Baseline - sklearn implementation
km = KMeans(n_clusters=2, n_init=1, random_state=98765).fit(X)
print("Sklearn K-means NMI:", nmi(y.numpy(), km.labels_))

simulator = GossipSimulator(
    data_dispatcher=DataDispatcher(data_handler, eval_on_user=False),
    delta=1000,
    protocol=AntiEntropyProtocol.PUSH,
    gossip_node_class=GossipNode,
    gossip_node_params={},
    model_handler_class=KMeansHandler,
    model_handler_params={
        "k" : 2,
        "dim" : data_handler.size(1),
        "alpha" : 0.1,
        "matching" : "hungarian",
        "create_model_mode" : CreateModelMode.MERGE_UPDATE
    },
    topology=None,
    #delay=(1, 4),
    #online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
    #drop_prob=.5, #Simulates the possibility of message dropping
    sampling_eval=0.01,
    round_synced=True
)

simulator.init_nodes()
evaluation, evaluation_user = simulator.start(n_rounds=100)

print(simulator.nodes[0].model_handler.model)

plot_evaluation([evaluation])
plot_evaluation([evaluation_user])


