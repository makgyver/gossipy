from gossipy.utils import plot_evaluation
from gossipy.simul import GossipSimulator, SimulationReport
from byzantine_report import ByzantineSimulationReport
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
import pickle


# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


class KMeansSameValueAttackHandler(SameValueAttackMixin, KMeansHandler):
    pass


# Baseline - the sklearn implementation is highly unstable in terms of NMI

def kmeans(X, k=2, max_iterations=400):
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
    for i in range(max_iterations):
        centroids = np.vstack([X[P == i, :].mean(axis=0)
                               for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
        if np.array_equal(P, tmp):
            break
        P = tmp
    return P


def eval(data_in, prop):
    if not data_in:
        plus = "+"
    else:
        plus = ""

    set_seed(42)
    X, y = load_classification_dataset(
        "spambase", normalize=True, as_tensor=True)
    data_handler = ClusteringDataHandler(X, y)

    P = kmeans(X.numpy(), k=2)
    print("K-means NMI:", nmi(y.numpy(), P))

    # Baseline - sklearn implementation
    km = KMeans(n_clusters=2, n_init=1, random_state=98765).fit(X)
    print("Sklearn K-means NMI:", nmi(y.numpy(), km.labels_))

    data_dispatcher = DataDispatcher(
        data_handler, eval_on_user=False, auto_assign=True)

    if data_in:
        total_nb = data_dispatcher.size()
        malicious_nb = ceil(data_dispatcher.size() * prop)
        normal_nb = total_nb - malicious_nb
    else:
        total_nb = ceil(data_dispatcher.size() * (1. + prop))
        malicious_nb = total_nb - data_dispatcher.size()
        normal_nb = data_dispatcher.size()

    nodes = generate_nodes(cls=GossipNode,
                           data_dispatcher=data_dispatcher,
                           p2p_net=StaticP2PNetwork(
                               total_nb),
                           model_proto=[(normal_nb, KMeansHandler(
                               k=2,
                               dim=data_handler.size(1),
                               alpha=0.1,
                               matching="hungarian",
                               create_model_mode=CreateModelMode.MERGE_UPDATE)),
                               (malicious_nb, KMeansSameValueAttackHandler(
                                k=2,
                                dim=data_handler.size(1),
                                alpha=0.1,
                                matching="hungarian",
                                create_model_mode=CreateModelMode.MERGE_UPDATE), data_in)],
                           round_len=1000,
                           sync=True)

    simulator = GossipSimulator(
        nodes=nodes,
        data_dispatcher=data_dispatcher,
        delta=1000,
        protocol=AntiEntropyProtocol.PUSH,
        #delay=UniformDelay(1, 4),
        # online_prob=.2, #Approximates the average online rate of the STUNner's smartphone traces
        # drop_prob=.1,  # Simulates the possibility of message dropping
        sampling_eval=0.01
    )

    report = ByzantineSimulationReport()
    simulator.add_receiver(report)
    simulator.init_nodes(seed=42)
    simulator.start(n_rounds=1000)

    if not data_in:
        plus = "+"
    else:
        plus = ""

    with open("../berta_{strplus}{nb:.0f}%-samevalue.pickle".format(strplus=plus, nb=prop*100.), "wb") as file:
        pickle.dump([[ev for _, ev in report.get_evaluation(False)]], file)

    plot_evaluation([[ev for _, ev in report.get_evaluation(False)]],
                    "Overall test results", "../berta_{strplus}{nb:.0f}%-samevalue.png".format(strplus=plus, nb=prop*100.))


for i in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  # 0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3,
    for val in [True, False]:
        eval(val, i)
