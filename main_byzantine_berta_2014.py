from gossipy.utils import plot_evaluation
from gossipy.simul import SimulationReport
from byzantine_report import GossipSimulatorByzantine
from gossipy.data.handler import ClusteringDataHandler
from gossipy.data import load_classification_dataset, DataDispatcher
from gossipy.model.handler import KMeansHandler
from gossipy.node import GossipNode
from byzantine_generate import generate_nodes
from math import ceil, floor
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork, UniformDelay
from gossipy import set_seed
from byzantine_handler import SameValueAttackMixin, AvoidReportMixin
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np


# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


class KMeansSameValueAttackHandler(KMeansHandler, SameValueAttackMixin, AvoidReportMixin):
    pass


set_seed(42)
X, y = load_classification_dataset("spambase", normalize=True, as_tensor=True)
data_handler = ClusteringDataHandler(X, y)

# Baseline - the sklearn implementation is highly unstable in terms of NMI


def kmeans(X, k=2, max_iterations=400):
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
    for i in range(max_iterations):
        centroids = np.vstack([X[P == i, :].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
        if np.array_equal(P, tmp):
            break
        P = tmp
    return P


P = kmeans(X.numpy(), k=2)
print("K-means NMI:", nmi(y.numpy(), P))

# Baseline - sklearn implementation
km = KMeans(n_clusters=2, n_init=1, random_state=98765).fit(X)
print("Sklearn K-means NMI:", nmi(y.numpy(), km.labels_))

data_dispatcher = DataDispatcher(
    data_handler, eval_on_user=False, auto_assign=True)

nodes = generate_nodes(cls=GossipNode,
                       data_dispatcher=data_dispatcher,
                       p2p_net=StaticP2PNetwork(data_dispatcher.size()),
                       model_proto=[(ceil(1 * data_dispatcher.size()), KMeansHandler(
                           k=2,
                           dim=data_handler.size(1),
                           alpha=0.1,
                           matching="hungarian",
                           create_model_mode=CreateModelMode.MERGE_UPDATE)),
                           (floor(0 * data_dispatcher.size()), KMeansSameValueAttackHandler(
                               k=2,
                               dim=data_handler.size(1),
                               alpha=0.1,
                               matching="hungarian",
                               create_model_mode=CreateModelMode.MERGE_UPDATE))],
                       round_len=1000,
                       sync=True)

simulator = GossipSimulatorByzantine(
    nodes=nodes,
    data_dispatcher=data_dispatcher,
    delta=1000,
    protocol=AntiEntropyProtocol.PUSH,
    #delay=UniformDelay(1, 4),
    # online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
    drop_prob=.1,  # Simulates the possibility of message dropping
    sampling_eval=0.01
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=100)

plot_evaluation([[ev for _, ev in report.get_evaluation(False)]],
                "Overall test results")
