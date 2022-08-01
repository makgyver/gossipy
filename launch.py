from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
import typer
from os import makedirs
import pickle
import json

from math import ceil

import torch
import networkx as nx
from networkx.generators import barabasi_albert_graph, erdos_renyi_graph, random_tree

from gossipy import set_seed
from gossipy.data import load_classification_dataset, get_CIFAR10, get_FashionMNIST, get_MNIST, DataDispatcher
from gossipy.data.handler import ClassificationDataHandler, ClusteringDataHandler, RegressionDataHandler
from gossipy.core import CreateModelMode, StaticP2PNetwork, ConstantDelay, UniformDelay, LinearDelay
from gossipy.model.handler import KMeansHandler, AdaLineHandler, PegasosHandler, TorchModelHandler, SamplingTMH, PartitionedTMH
from gossipy.model.nn import AdaLine, LogisticRegression, TorchMLP
from gossipy.node import GossipNode, PassThroughNode, CacheNeighNode, SamplingBasedNode, PartitioningBasedNode, PENSNode
from gossipy.simul import GossipSimulator, TokenizedGossipSimulator, AntiEntropyProtocol
from gossipy.utils import plot_evaluation

from byzantine_generate import GenerationType, generate_nodes
from byzantine_handler import TorchModelRandomGradientAttackHandler, TorchModelSameValueAttackHandler, TorchModelGradientScalingAttackHandler, TorchModelBackGradientAttackHandler, TorchModelRandomModelAttackHandler, TorchModelRandomFullModelAttackHandler, PegasosRandomGradientAttackHandler, PegasosSameValueAttackHandler, PegasosGradientScalingAttackHandler, PegasosBackGradientAttackHandler, PegasosRandomModelAttackHandler, PegasosRandomFullModelAttackHandler
from byzantine_report import ByzantineSimulationReport
from byzantine_network import multi_clique_network, ring_network

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "Apache, Version 2.0"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

app = typer.Typer(add_completion=False)


class ModelHandlerEnum(str, Enum):
    KMEANS = "kmeans"
    ADALINE = "adaline"
    PEGASOS = "pegasos"
    TORCH_MODEL = "torch_model"
    SAMPLING_TMH = "sampling_tmh"
    PARTITIONED_TMH = "partitioned_tmh"

    def __str__(self):
        return self.value


class KMeansMatchingEnum(str, Enum):
    NAIVE = "naive"
    HUNGARIAN = "hungarian"

    def __str__(self):
        return self.value


class TorchNetEnum(str, Enum):
    LINEAR = "linear"
    MLP = "mlp"
    MLP_SOFTMAX = "mlp_softmax"
    CONVOLUTIONAL = "convolutional"

    def __str__(self):
        return self.value


class TorchActivationEnum(str, Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"

    def __str__(self):
        return self.value

    def to_activation_function(self):
        if self == TorchActivationEnum.SIGMOID:
            return torch.nn.Sigmoid
        elif self == TorchActivationEnum.RELU:
            return torch.nn.ReLU
        else:
            raise RuntimeError("Wrong value for TorchActivationEnum.")


class TorchCriterionEnum(str, Enum):
    BCE = "bce"
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"

    def __str__(self):
        return self.value

    def to_loss_function(self):
        if self == TorchCriterionEnum.BCE:
            return torch.nn.BCE()
        elif self == TorchCriterionEnum.CROSS_ENTROPY:
            return torch.nn.CrossEntropyLoss()
        elif self == TorchCriterionEnum.MSE:
            return torch.nn.MSELoss()
        else:
            raise RuntimeError("Wrong value for TorchCriterionEnum.")


class CreateModelModeEnum(str, Enum):
    UPDATE = "update"
    MERGE_UPDATE = "merge_update"
    UPDATE_MERGE = "update_merge"
    PASS = "pass"

    def __str__(self):
        return self.value

    def to_create_model_mode(self):
        if self == CreateModelModeEnum.UPDATE:
            return CreateModelMode.UPDATE
        elif self == CreateModelModeEnum.MERGE_UPDATE:
            return CreateModelMode.MERGE_UPDATE
        elif self == CreateModelModeEnum.UPDATE_MERGE:
            return CreateModelMode.UPDATE_MERGE
        elif self == CreateModelModeEnum.PASS:
            return CreateModelMode.PASS
        else:
            raise RuntimeError("Wrong value in CreateModelModeEnum.")


class DataHandlerEnum(str, Enum):
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    REGRESSION = "regression"

    def __str__(self):
        return self.value


class NodeEnum(str, Enum):
    GOSSIP = "gossip"
    PASSTROUGH = "passthrough"
    CACHE_NEIGH = "cache_neigh"
    SAMPLING_BASED = "sampling_based"
    PARTITIONING_BASED = "partitioning_based"
    PENS = "pens"

    def __str__(self):
        return self.value

    def to_node_class(self):
        if self == NodeEnum.GOSSIP:
            return GossipNode
        elif self == NodeEnum.PASSTROUGH:
            return PassThroughNode
        elif self == NodeEnum.CACHE_NEIGH:
            return CacheNeighNode
        elif self == NodeEnum.SAMPLING_BASED:
            return SamplingBasedNode
        elif self == NodeEnum.PARTITIONING_BASED:
            return PartitioningBasedNode
        elif self == NodeEnum.PENS:
            return PENSNode
        else:
            raise RuntimeError("Wrong value for NodeEnum.")


class SimulatorEnum(str, Enum):
    GOSSIP = "gossip"
    TOKENIZED = "tokenized"

    def __str__(self):
        return self.value


class AntiEntropyProtocolEnum(str, Enum):
    PUSH = "push"
    PULL = "pull"
    PUSH_PULL = "push_pull"

    def __str__(self):
        return self.value

    def to_anti_entropy_protocol(self):
        if self == AntiEntropyProtocolEnum.PUSH:
            return AntiEntropyProtocol.PUSH
        elif self == AntiEntropyProtocolEnum.PULL:
            return AntiEntropyProtocol.PULL
        elif self == AntiEntropyProtocolEnum.PUSH_PULL:
            return AntiEntropyProtocol.PUSH_PULL
        else:
            raise RuntimeError("Wrong value for AntiProtocolEnum.")


class DataSetEnum(str, Enum):
    IRIS = "iris"
    BREAST = "breast"
    DIGITS = "digits"
    WINE = "wine"
    REUTERS = "reuters"
    SONAR = "sonar"
    INOSPHERE = "ionosphere"
    ABALONE = "abalone"
    BANKNOTE = "banknote"
    SPAMBASE = "spambase"
    # RECSYS = "recsys"
    CIFAR10 = "cifar10"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    # FEMNIST = "femnist"
    PATH = "path"

    def __str__(self):
        return self.value


class TopologyEnum(str, Enum):
    PLAIN = "plain"
    RANDOM_TREE = "random_tree"
    # Binomial graph
    ERDOS_RENYI = "erdos_renyi"
    # Power law graph
    BARABASI_ALBERT = "barabasi_albert"
    RING = "ring"
    CLIQUES = "cliques"

    def __str__(self):
        return self.value


class AttackEnum(str, Enum):
    GRADIENT_SCALING = "gradient_scaling"
    BACKGRADIENT = "backgradient"
    SAME_VALUE = "same_value"
    RANDOM_GRADIENT = "random_gradient"
    RANDOM_MODEL = "random_model"
    RANDOM_FULL_MODEL = "random_full_model"
    # LEARNING = "learning"

    def __str__(self):
        return self.value


class GenerationTypeEnum(str, Enum):
    NORMAL = "normal"
    SHUFFLED = "shuffled"
    LOW_DEGREE = "low_degree"
    HIGH_DEGREE = "high_degree"

    def __str__(self):
        return self.value

    def to_generation_type(self):
        if self == GenerationTypeEnum.NORMAL:
            return GenerationType.NORMAL
        elif self == GenerationTypeEnum.SHUFFLED:
            return GenerationType.SHUFFLED
        elif self == GenerationTypeEnum.LOW_DEGREE:
            return GenerationType.LOW_DEGREE
        elif self == GenerationTypeEnum.HIGH_DEGREE:
            return GenerationType.HIGH_DEGREE
        else:
            raise RuntimeError("Wrong value for GenerationTypeEnum.")


class DelayEnum(str, Enum):
    CONSTANT = "constant"
    UNIFORM = "uniform"
    LINEAR = "linear"

    def __str__(self):
        return self.value


# TODO : Case insensitive choice


@app.command()
def run(model_handler_type: ModelHandlerEnum = typer.Option(ModelHandlerEnum.TORCH_MODEL, help="Model handler choosen for this simulation, some handlers require other options (torch_*, kmeans_*, n_samples, sample_size...)", case_sensitive=False),
        create_model_mode: CreateModelModeEnum = typer.Option(
            CreateModelModeEnum.MERGE_UPDATE, help="CreateModelMode for this simulation, change only if needed. Changing it will malicious nodes can lead to strange behavior (malicious nodes may run normal updates and vice-versa.)", case_sensitive=False),
        torch_net: Optional[TorchNetEnum] = typer.Option(
            None, help="Pytorch network template to use with a torch_model handler (or sampling_tmh and partitioned_tmh)", case_sensitive=False),
        torch_activation: Optional[TorchActivationEnum] = typer.Option(
            None, help="Pytorch activation function to use with mlp or mlp_softmax torch model", case_sensitive=False),
        torch_criterion: Optional[TorchCriterionEnum] = typer.Option(
            None, help="Pytorch criterion to use with a torch_model handler (or sampling_tmh and partitioned_tmh)", case_sensitive=False),
        torch_local_epochs: int = typer.Option(
            1, help="Number of training with local data per iteration with a torch_model handler (or sampling_tmh and partitioned_tmh)", case_sensitive=False),
        torch_batch_size: int = typer.Option(
            32, help="Batch size with a torch_model handler (or sampling_tmh and partitioned_tmh)", case_sensitive=False),
        torch_hidden_layer_dim: List[int] = typer.Option(
            [], help="Number of neuron per layer. Use it with a mlp or mlp_sofmax torch network. You can use this option multiple times if you want multiple hidden layers.", case_sensitive=False),
        learning_rate: float = typer.Option(
            0.01, help="Learning rate of the mode (works with almost all handlers.)", case_sensitive=False),
        weight_decay: float = typer.Option(
            0, help="Weight decay for torch model handlers", case_sensitive=False),
        sample_size: Optional[float] = typer.Option(
            None, help="Sample size of sampling tmh.", case_sensitive=False),
        n_parts: Optional[int] = typer.Option(
            None, help="Number of partition in partitionned tmh.", case_sensitive=False),
        kmeans_k: Optional[int] = typer.Option(
            None, help="Number of points in k-means model.", case_sensitive=False),
        kmeans_alpha: float = typer.Option(
            0.1, help="Leaning coefficient of kmeans model", case_sensitive=False),
        kmeans_matching: KMeansMatchingEnum = typer.Option(
            KMeansMatchingEnum.NAIVE, help="Type of matching used for k-means learning", case_sensitive=False),
        dataset_name: DataSetEnum = typer.Option(
            DataSetEnum.SPAMBASE, help="NAme of the used dataset", case_sensitive=False),
        dataset_path: Optional[Path] = typer.Option(
            None, help="If dataset_name if path, use this options to give path of file", case_sensitive=False),
        dataset_normalize: bool = typer.Option(
            True, help="Wether data should be normalized", case_sensitive=False),
        dataset_widerange: Optional[bool] = typer.Option(
            None, help="Wether data targets/labels should be between [0, 1] or [-1, 1], do not specify it to let the launcher choose with model_handle", case_sensitive=False),
        dataset_test_proportion: float = typer.Option(
            0.1, help="Dataset training set proportion kept aside for testing purpose, if dataset_use_test is set to True and the dataset has a testing set, this option is ignored.", case_sensitive=False),
        dataset_use_test: bool = typer.Option(
            True, help="Whether the test set of the dataset should be used, if any", case_sensitive=False),
        data_handler_type: DataHandlerEnum = typer.Option(
            DataHandlerEnum.CLASSIFICATION, help="Type of data handler", case_sensitive=False),
        topology_type: TopologyEnum = typer.Option(
            TopologyEnum.PLAIN, help="Topology of the used network. Some topologies need more options.", case_sensitive=False),
        barabasi_m: Optional[int] = typer.Option(
            None, help="m value of barabasi topology", case_sensitive=False),
        erdos_renyi_prob: Optional[float] = typer.Option(
            None, help="Probability for Erdos-Renyi topology", case_sensitive=False),
        clique_size: Optional[List[int]] = typer.Option(
            None, help="Size of cliques for clique topology.\nIf ther is only one number, there will be as much as possible topology with this size and the later will take the rest of nodes.\nIf more than one value is given, only one clique of each value will be created. Works only if total matches the number of nodes.", case_sensitive=False),
        n_nodes: Optional[int] = typer.Option(
            None, help="Number of nodes", case_sensitive=False),
        n_rounds: int = typer.Option(
            150, help="Number of simulated rounds", case_sensitive=False),
        node_type: NodeEnum = typer.Option(
            NodeEnum.GOSSIP, help="Type of node", case_sensitive=False),
        sync_nodes: bool = typer.Option(
            False, help="Wether nodes should be synced. If you don't know leave it as it", case_sensitive=False),
        round_len: int = typer.Option(
            100, help="Length of a round", case_sensitive=False),
        pens_n_sample: int = typer.Option(
            10, help="For PENS nodes, number of sampels", case_sensitive=False),
        pens_m_top: int = typer.Option(
            2, help="For PENS nodes, value of m_top", case_sensitive=False),
        pens_step1_rounds: int = typer.Option(
            100, help="For PENS nodes, length of step 1 in rouns", case_sensitive=False),
        generation_type: GenerationTypeEnum = typer.Option(
            GenerationTypeEnum.NORMAL, help="Type of generation (how malicious and normal behaviors are allocated to nodes. Ex: low => normal nodes are allocated on low degree nodes and malicious nodes on high degree nodes)", case_sensitive=False),
        simulator_type: SimulatorEnum = typer.Option(
            SimulatorEnum.GOSSIP, help="Type of simulator, for now, only gossip is supported by this launcher", case_sensitive=False),
        delta: int = typer.Option(100, help="Delta", case_sensitive=False),
        anti_entropy_protocol: AntiEntropyProtocolEnum = typer.Option(
            AntiEntropyProtocolEnum.PUSH, help="Anti Entropy Protocol, leave it as it if you don't know.", case_sensitive=False),
        drop_prob: float = typer.Option(
            0., help="Probability of a packet to be lost in network", case_sensitive=False),
        online_prob: float = typer.Option(
            1., help="Probability of a node to be online at each round", case_sensitive=False),
        delay_mode: Optional[DelayEnum] = typer.Option(
            None, help="Type of delay (related to network packet speed)", case_sensitive=False),
        delay_constant_value: Optional[int] = typer.Option(
            None, help="Length of delay for constant delay or fixed part of linear delays.", case_sensitive=False),
        delay_min: Optional[int] = typer.Option(
            None, help="Minimum delay for uniform delay", case_sensitive=False),
        delay_max: Optional[int] = typer.Option(
            None, help="Maximum delay for uniform delay", case_sensitive=False),
        delay_factor: Optional[float] = typer.Option(
            None, help="Multiplication factor for linear delay (depends on packet size/length)", case_sensitive=False),
        sampling_eval: float = typer.Option(0., help="Proportion of nodes evaluated to evaluate the whole set (if 0, all the nodes are evaluated.)\nA low value speeds up simulation by reducing time taken by statistics but may lead variability in results.\nEach malicious returns a None value in evaluation so, if there is too much malicious nodes and this value is too low, statistics may be useless.", case_sensitive=False),
        attack_type: Optional[AttackEnum] = typer.Option(
            None, help="Type of malicious nodes. Leave it if you don't want to try any byzantine attack and you just want a simulation", case_sensitive=False),
        attack_proportion: Optional[float] = typer.Option(
            None, help="Proportion of malicious nodes", case_sensitive=False),
        attack_own_data: bool = typer.Option(
            True, help="Wether malicious nodes own data or they are added to the simulation", case_sensitive=False),
        attack_scale: Optional[float] = typer.Option(
            None, help="Scale parameters for attacks (depends on attack type)", case_sensitive=False),
        attack_mean: Optional[float] = typer.Option(
            None, help="Mean parameters for attacks (depends on attack type)", case_sensitive=False),
        attack_pegasos_nb: int = typer.Option(
            1, help="Value added to n_update whith a pegasos attack", case_sensitive=False),
        seed: int = typer.Option(
            42, help="Seed of Random Number Generators set a the starting of the program", case_sensitive=False),
        on_device: bool = typer.Option(
            True, help="Whether simulation is calculated on a compatible GPU if any", case_sensitive=False),
        result_folder: str = typer.Option(
            "results", help="Parent folder of results", case_sensitive=False),
        simul_name: Optional[str] = typer.Option(None, help="Name of the simulation for results (can be used to reduce the length of files, be carefull of doublons...)", case_sensitive=False)):
    run_impl(model_handler_type, create_model_mode, torch_net, torch_activation, torch_criterion, torch_local_epochs, torch_batch_size, torch_hidden_layer_dim, learning_rate, weight_decay, sample_size, n_parts, kmeans_k, kmeans_alpha, kmeans_matching, dataset_name, dataset_path, dataset_normalize, dataset_widerange, dataset_test_proportion, dataset_use_test, data_handler_type, topology_type, barabasi_m, erdos_renyi_prob, clique_size,
             n_nodes, n_rounds, node_type, sync_nodes, round_len, pens_n_sample, pens_m_top, pens_step1_rounds, generation_type, simulator_type, delta, anti_entropy_protocol, drop_prob, online_prob, delay_mode, delay_constant_value, delay_min, delay_max, delay_factor, sampling_eval, attack_type, attack_proportion, attack_own_data, attack_scale, attack_mean, attack_pegasos_nb, seed, on_device, result_folder, simul_name)


def run_impl(model_handler_type: ModelHandlerEnum = ModelHandlerEnum.TORCH_MODEL,
             create_model_mode: CreateModelModeEnum = CreateModelModeEnum.MERGE_UPDATE,
             torch_net: Optional[TorchNetEnum] = None,
             torch_activation: Optional[TorchActivationEnum] = None,
             torch_criterion: Optional[TorchCriterionEnum] = None,
             torch_local_epochs: int = 1,
             torch_batch_size: int = 32,
             torch_hidden_layer_dim: List[int] = [],
             learning_rate: float = 0.01,
             weight_decay: float = 0,
             sample_size: Optional[float] = None,
             n_parts: Optional[int] = None,
             kmeans_k: Optional[int] = None,
             kmeans_alpha: float = 0.1,
             kmeans_matching: KMeansMatchingEnum = KMeansMatchingEnum.NAIVE,
             dataset_name: DataSetEnum = DataSetEnum.SPAMBASE,
             dataset_path: Optional[Path] = None,
             dataset_normalize: bool = True,
             dataset_widerange: Optional[bool] = None,
             dataset_test_proportion: float = 0.1,
             dataset_use_test: bool = True,
             data_handler_type: DataHandlerEnum = DataHandlerEnum.CLASSIFICATION,
             topology_type: TopologyEnum = TopologyEnum.PLAIN,
             barabasi_m: Optional[int] = None,
             erdos_renyi_prob: Optional[float] = None,
             clique_size: Optional[List[int]] = None,
             n_nodes: Optional[int] = None,
             n_rounds: int = 150,
             node_type: NodeEnum = NodeEnum.GOSSIP,
             sync_nodes: bool = False,
             round_len: int = 100,
             pens_n_sample: int = 10,
             pens_m_top: int = 2,
             pens_step1_rounds: int = 100,
             generation_type: GenerationTypeEnum = GenerationTypeEnum.NORMAL,
             simulator_type: SimulatorEnum = SimulatorEnum.GOSSIP,
             delta: int = 100,
             anti_entropy_protocol: AntiEntropyProtocolEnum = AntiEntropyProtocolEnum.PUSH,
             drop_prob: float = 0.,
             online_prob: float = 1.,
             delay_mode: Optional[DelayEnum] = None,
             delay_constant_value: Optional[int] = None,
             delay_min: Optional[int] = None,
             delay_max: Optional[int] = None,
             delay_factor: Optional[float] = None,
             sampling_eval: float = 0.,
             attack_type: Optional[AttackEnum] = None,
             attack_proportion: Optional[float] = None,
             attack_own_data: bool = True,
             attack_scale: Optional[float] = None,
             attack_mean: Optional[float] = None,
             attack_pegasos_nb: int = 1,
             seed: int = 42,
             on_device: bool = True,
             result_folder: str = "results",
             simul_name: Optional[str] = None):
    verify_args(model_handler_type, create_model_mode, torch_net, torch_activation, torch_criterion, torch_local_epochs, torch_batch_size, torch_hidden_layer_dim, learning_rate, weight_decay, sample_size, n_parts, kmeans_k, kmeans_alpha, kmeans_matching, dataset_name, dataset_path, dataset_normalize, dataset_widerange, dataset_test_proportion, dataset_use_test, data_handler_type, topology_type, barabasi_m, erdos_renyi_prob, clique_size,
                n_nodes, n_rounds, node_type, sync_nodes, round_len, pens_n_sample, pens_m_top, pens_step1_rounds, generation_type, simulator_type, delta, anti_entropy_protocol, drop_prob, online_prob, delay_mode, delay_constant_value, delay_min, delay_max, delay_factor, sampling_eval, attack_type, attack_proportion, attack_own_data, attack_scale, attack_mean, attack_pegasos_nb, seed, on_device, result_folder, simul_name)

    local_args = save_args(model_handler_type=model_handler_type, create_model_mode=create_model_mode, torch_net=torch_net, torch_activation=torch_activation, torch_criterion=torch_criterion, torch_local_epochs=torch_local_epochs, torch_batch_size=torch_batch_size, torch_hidden_layer_dim=torch_hidden_layer_dim, learning_rate=learning_rate, weight_decay=weight_decay, sample_size=sample_size, n_parts=n_parts, kmeans_k=kmeans_k, kmeans_alpha=kmeans_alpha, kmeans_matching=kmeans_matching, dataset_name=dataset_name, dataset_path=dataset_path, dataset_normalize=dataset_normalize, dataset_widerange=dataset_widerange, dataset_test_proportion=dataset_test_proportion, dataset_use_test=dataset_use_test, data_handler_type=data_handler_type, topology_type=topology_type, barabasi_m=barabasi_m, erdos_renyi_prob=erdos_renyi_prob, clique_size=clique_size,
                           n_nodes=n_nodes, n_rounds=n_rounds, node_type=node_type, sync_nodes=sync_nodes, round_len=round_len, pens_n_sample=pens_n_sample, pens_m_top=pens_m_top, pens_step1_rounds=pens_step1_rounds, generation_type=generation_type, simulator_type=simulator_type, delta=delta, anti_entropy_protocol=anti_entropy_protocol, drop_prob=drop_prob, online_prob=online_prob, delay_mode=delay_mode, delay_constant_value=delay_constant_value, delay_min=delay_min, delay_max=delay_max, delay_factor=delay_factor, sampling_eval=sampling_eval, attack_type=attack_type, attack_proportion=attack_proportion, attack_own_data=attack_own_data, attack_scale=attack_scale, attack_mean=attack_mean, attack_pegasos_nb=attack_pegasos_nb, seed=seed, on_device=on_device, result_folder=result_folder, simul_name=simul_name)

    # Random seed
    set_seed(seed)

    # Loading dataset in X_tr, y_tr, X_te and y_te
    if (dataset_name == DataSetEnum.CIFAR10):
        train_set, test_set = get_CIFAR10()
        X_tr = train_set[0]
        y_tr = train_set[1]
        if dataset_use_test:
            X_te = test_set[0]
            y_te = test_set[1]
        else:
            X_te = None
            y_te = None
    elif (dataset_name == DataSetEnum.MNIST):
        train_set, test_set = get_MNIST(
            normalize=dataset_normalize)
        X_tr = train_set[0]
        y_tr = train_set[1]
        if dataset_use_test:
            X_te = test_set[0]
            y_te = test_set[1]
        else:
            X_te = None
            y_te = None
    elif (dataset_name == DataSetEnum.FASHION_MNIST):
        train_set, test_set = get_FashionMNIST()
        X_tr = train_set[0]
        y_tr = train_set[1]
        if dataset_use_test:
            X_te = test_set[0]
            y_te = test_set[1]
        else:
            X_te = None
            y_te = None
    elif dataset_name == DataSetEnum.PATH:
        X_tr, y_tr = load_classification_dataset(
            dataset_path, normalize=dataset_normalize)
        X_te, y_te = None, None
    else:
        X_tr, y_tr = load_classification_dataset(
            dataset_name.value, normalize=dataset_normalize)
        X_te, y_te = None, None

    # Changing range from [0;1] to [-1;-1]
    # We keep dataset_widerange for naming after
    if dataset_widerange is None:
        dataset_wide = model_handler_type == ModelHandlerEnum.PEGASOS or model_handler_type == ModelHandlerEnum.ADALINE
    else:
        dataset_wide = dataset_widerange

    if dataset_wide == True:
        y_tr = 2*y_tr - 1
        if y_te:
            y_te = 2*y_te - 1

    # Setting data handler and dispatcher
    if data_handler_type == DataHandlerEnum.CLASSIFICATION:
        data_handler = ClassificationDataHandler(
            X_tr, y_tr, X_te, y_te, 0. if X_te is not None else dataset_test_proportion, seed, on_device=on_device)
    elif data_handler_type == DataHandlerEnum.CLUSTERING:
        data_handler = ClusteringDataHandler(
            X_tr, y_tr, on_device=on_device)
    elif data_handler == DataHandlerEnum.REGRESSION:
        data_handler = RegressionDataHandler(
            X_tr, y_tr, X_te, y_te, 0. if X_te is not None else dataset_test_proportion, seed, on_device=on_device)
    else:
        raise RuntimeError("Wrong value for data_handler_type")

    data_dispatcher = DataDispatcher(
        data_handler, data_handler.size() if n_nodes is None else n_nodes, eval_on_user=False, auto_assign=True)

    # Normal Model Handler
    if model_handler_type == ModelHandlerEnum.KMEANS:
        model_handler = KMeansHandler(
            kmeans_k,
            data_handler.size(1),
            kmeans_alpha,
            kmeans_matching,
            create_model_mode.to_create_model_mode())
    elif model_handler_type == ModelHandlerEnum.ADALINE:
        net = AdaLine(data_handler.size(1))
        model_handler = AdaLineHandler(
            net,
            learning_rate,
            create_model_mode.to_create_model_mode(),
            True,
            on_device)
    elif model_handler_type == ModelHandlerEnum.PEGASOS:
        net = AdaLine(data_handler.size(1))
        model_handler = PegasosHandler(
            net,
            learning_rate,
            create_model_mode.to_create_model_mode(),
            True,
            on_device)
    else:
        if torch_net == TorchNetEnum.LINEAR:
            net = LogisticRegression(
                data_handler.size(1), data_handler.n_classes)
        elif torch_net == TorchNetEnum.MLP:
            net = TorchMLP(data_handler.size(
                1),
                data_handler.n_classes,
                torch_hidden_layer_dim,
                torch_activation.to_activation_function(),
                False
            )
        elif torch_net == TorchNetEnum.MLP_SOFTMAX:
            net = TorchMLP(data_handler.size(
                1),
                data_handler.n_classes,
                torch_hidden_layer_dim,
                torch_activation.to_activation_function(),
                False
            )
        else:
            raise RuntimeError("Wrong value for torch_net.")

        criterion = torch_criterion.to_loss_function()

        if model_handler_type == ModelHandlerEnum.TORCH_MODEL:
            model_handler = TorchModelHandler(
                net,
                optimizer=torch.optim.SGD,
                optimizer_params={
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                },
                criterion=criterion,
                local_epochs=torch_local_epochs,
                batch_size=torch_batch_size,
                create_model_mode=create_model_mode.to_create_model_mode(),
                copy_model=True,
                on_device=on_device)
        elif model_handler_type == ModelHandlerEnum.SAMPLING_TMH:
            model_handler = SamplingTMH(
                sample_size,
                net,
                optimizer=torch.optim.SGD,
                optimizer_params={
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                },
                criterion=criterion,
                local_epochs=torch_local_epochs,
                batch_size=torch_batch_size,
                create_model_mode=create_model_mode.to_create_model_mode(),
                copy_model=True,
                on_device=on_device)
        elif model_handler_type == ModelHandlerEnum.PARTITIONED_TMH:
            model_handler = PartitionedTMH(
                net,
                TorchModelPartition(net, n_parts),
                optimizer=torch.optim.SGD,
                optimizer_params={
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                },
                criterion=criterion,
                local_epochs=torch_local_epochs,
                batch_size=torch_batch_size,
                create_model_mode=create_model_mode.to_create_model_mode(),
                copy_model=True,
                on_device=on_device)
        else:
            raise RuntimeError("Wrong value for model_handler.")

    # Malicious node, if any
    if attack_type is not None:
        if model_handler_type == ModelHandlerEnum.PEGASOS:
            handler_args = (net,
                            learning_rate,
                            create_model_mode.to_create_model_mode(),
                            True,
                            on_device)
            if attack_type == AttackEnum.SAME_VALUE:
                attack_model_handler = PegasosSameValueAttackHandler(
                    *handler_args)
            elif attack_type == AttackEnum.BACKGRADIENT:
                attack_model_handler = PegasosBackGradientAttackHandler(
                    attack_pegasos_nb, *handler_args)
            elif attack_type == AttackEnum.GRADIENT_SCALING:
                attack_model_handler = PegasosGradientScalingAttackHandler(
                    attack_scale, attack_pegasos_nb, *handler_args)
            elif attack_type == AttackEnum.RANDOM_GRADIENT:
                attack_model_handler = PegasosRandomGradientAttackHandler(
                    attack_scale, attack_pegasos_nb, *handler_args)
            elif attack_type == AttackEnum.RANDOM_MODEL:
                attack_model_handler = PegasosRandomModelAttackHandler(
                    attack_pegasos_nb, *handler_args)
            elif attack_type == AttackEnum.RANDOM_FULL_MODEL:
                attack_model_handler = PegasosRandomFullModelAttackHandler(
                    attack_scale, attack_mean, attack_pegasos_nb, *handler_args)
        elif model_handler_type == ModelHandlerEnum.TORCH_MODEL:
            handler_args = (net,
                            torch.optim.SGD,
                            {
                                "lr": learning_rate,
                                "weight_decay": weight_decay
                            },
                            criterion,
                            torch_local_epochs,
                            torch_batch_size,
                            create_model_mode.to_create_model_mode(),
                            True,
                            on_device)
            if attack_type == AttackEnum.SAME_VALUE:
                attack_model_handler = TorchModelSameValueAttackHandler(
                    *handler_args)
            elif attack_type == AttackEnum.BACKGRADIENT:
                attack_model_handler = TorchModelBackGradientAttackHandler(
                    *handler_args)
            elif attack_type == AttackEnum.GRADIENT_SCALING:
                attack_model_handler = TorchModelGradientScalingAttackHandler(
                    attack_scale, *handler_args)
            elif attack_type == AttackEnum.RANDOM_GRADIENT:
                attack_model_handler = TorchModelRandomGradientAttackHandler(
                    attack_scale, *handler_args)
            elif attack_type == AttackEnum.RANDOM_MODEL:
                attack_model_handler = TorchModelRandomModelAttackHandler(
                    *handler_args)
            elif attack_type == AttackEnum.RANDOM_FULL_MODEL:
                attack_model_handler = TorchModelRandomFullModelAttackHandler(
                    attack_scale, attack_mean, *handler_args)
        else:
            raise RuntimeError(
                "Byzantine attack is not supported on this kind of handler for now.")
    else:
        attack_model_handler = None

    # Number of nodes (taking into account malicious nodes)
    if attack_type is not None:
        if attack_own_data:
            n_total = data_dispatcher.size()
            n_malicious = ceil(data_dispatcher.size() * attack_proportion)
            n_normal = n_total - n_malicious
        else:
            n_total = ceil(data_dispatcher.size() * (1. + attack_proportion))
            n_malicious = n_total - data_dispatcher.size()
            n_normal = data_dispatcher.size()
    else:
        n_total = data_dispatcher.size()
        n_malicious = 0
        n_normal = n_total

    # Topology
    if topology_type == TopologyEnum.PLAIN:
        topology = StaticP2PNetwork(n_total, None)
        topology_matrix = None
    else:
        if topology_type == TopologyEnum.RANDOM_TREE:
            topology_matrix = nx.to_numpy_matrix(
                random_tree(n_total))
        elif topology_type == TopologyEnum.BARABASI_ALBERT:
            topology_matrix = nx.to_numpy_matrix(
                barabasi_albert_graph(n_total, barabasi_m))
        elif topology_type == TopologyEnum.ERDOS_RENYI:
            topology_matrix = nx.to_numpy_matrix(
                erdos_renyi_graph(n_total, erdos_renyi_prob))
        elif topology_type == TopologyEnum.CLIQUES:
            if len(clique_size == 1):
                clique_size = clique_size[0]
            topology_matrix = multi_clique_network(n_total, clique_size)
        elif topology_type == TopologyEnum.RING:
            topology_matrix = ring_network(n_total)
        else:
            raise RuntimeError("Wrong value for topology_type.")

        topology = StaticP2PNetwork(
            n_total, topology=topology_matrix)

    # Nodes
    if node_type == NodeEnum.PENS:
        node_args = (pens_n_sample, pens_m_top, pens_step1_rounds)
    else:
        node_args = ()

    nodes = generate_nodes(
        node_type.to_node_class(),
        data_dispatcher,
        topology,
        ((n_normal, model_handler),
         (n_malicious, attack_model_handler, attack_own_data)),
        round_len,
        sync_nodes,
        generation_type.to_generation_type())

    # Delay
    if (delay_mode == DelayEnum.CONSTANT):
        delay = ConstantDelay(delay_constant_value)
    elif (delay_mode == DelayEnum.UNIFORM):
        delay = UniformDelay(delay_min, delay_max)
    elif (delay_mode == DelayEnum.LINEAR):
        delay = LinearDelay(delay_factor, delay_constant_value)
    else:
        delay = ConstantDelay(0)

    # Simulator
    if simulator_type == SimulatorEnum.GOSSIP:
        simulator = GossipSimulator(
            nodes,
            data_dispatcher,
            delta,
            anti_entropy_protocol.to_anti_entropy_protocol(),
            drop_prob,
            online_prob,
            delay,
            sampling_eval
        )
    elif simulator_type == SimulatorEnum.TOKENIZED:
        raise RuntimeError("Tokenized Simulation not managed for now.")
    else:
        raise RuntimeError("Wrong value for simulator_type")

    # Simulation and report
    report = ByzantineSimulationReport()
    simulator.add_receiver(report)
    simulator.init_nodes(seed=seed)
    simulator.start(n_rounds)

    # Saving
    if simul_name is None:
        if model_handler_type == ModelHandlerEnum.KMEANS:
            simul_name = model_handler_type + "-k" + str(kmeans_k) + \
                "-a" + str(kmeans_alpha) + "-" + str(kmeans_matching)
        elif model_handler_type == ModelHandlerEnum.ADALINE or model_handler_type == ModelHandlerEnum.PEGASOS:
            simul_name = model_handler_type + "-lr" + str(learning_rate)
        else:  # TorchModel or derivative
            simul_name = model_handler_type
            if (model_handler_type == ModelHandlerEnum.SAMPLING_TMH):
                simul_name += "-s" + str(sample_size)
            elif (model_handler_type == ModelHandlerEnum.PARTITIONED_TMH):
                simul_name += "-n" + str(n_parts)
            simul_name += "-" + torch_net
            if (torch_net == TorchNetEnum.MLP or torch_net == TorchNetEnum.MLP_SOFTMAX):
                for size in torch_hidden_layer_dim:
                    simul_name += "-" + str(size)
            simul_name += "-lr" + str(learning_rate)
            if (weight_decay != 0.):
                simul_name += "-wd" + str(weight_decay)
            simul_name += "-" + torch_criterion
            if (torch_local_epochs != 1):
                simul_name += "-le" + str(torch_local_epochs)
            if (torch_batch_size != 32):
                simul_name += "-bs" + str(torch_batch_size)
        if create_model_mode != CreateModelModeEnum.MERGE_UPDATE:
            simul_name += "-" + create_model_mode.lower()

        # WARN : For path dataset => custom
        simul_name += "-" + dataset_name
        if (dataset_normalize == False):
            simul_name += "_unnormalized"
        if (dataset_use_test == False):
            simul_name += "-notest"
        if (dataset_widerange == True):
            simul_name += "-wide"
        elif (dataset_widerange == False):
            simul_name += "-nowide"

        if (topology_type != TopologyEnum.PLAIN):
            simul_name += "-" + topology_type
            if (topology_type == TopologyEnum.BARABASI_ALBERT):
                simul_name += "-m" + str(barabasi_m)
            elif (topology_type == TopologyEnum.ERDOS_RENYI):
                simul_name += "-p" + str(erdos_renyi_prob)

        if (n_nodes is not None):
            simul_name += "-no" + str(n_nodes)

        if (node_type != NodeEnum.GOSSIP):
            simul_name += "-n" + node_type
            if (node_type == NodeEnum.PENS):
                simul_name += "-pn" + str(pens_n_sample)
                simul_name += "-pm" + str(pens_m_top)
                simul_name += "-ps" + str(pens_step1_rounds)

        if (sync_nodes):
            simul_name += "-sync"

        if (round_len != 100):
            simul_name += "-rl" + str(round_len)

        if (simulator_type != SimulatorEnum.GOSSIP):
            simul_name += "-" + simulator_type

        if (delta != 100):
            simul_name += "-d" + str(delta)

        if (anti_entropy_protocol != AntiEntropyProtocolEnum.PUSH):
            simul_name += "-" + anti_entropy_protocol.lower()

        if (drop_prob != 0. or online_prob != 1.):
            simul_name += "-dropout-dp" + \
                str(drop_prob) + "-op" + str(online_prob)

        if (delay_mode == DelayEnum.CONSTANT):
            simul_name += "-dconstant-" + str(delay_constant_value)
        elif (delay_mode == DelayEnum.UNIFORM):
            simul_name += "-duniform-" + str(delay_min) + "-" + str(delay_max)
        elif (delay_mode == DelayEnum.LINEAR):
            simul_name += "-dlinear-" + \
                str(delay_factor) + "-" + str(delay_constant_value)

    folder_name = result_folder

    if (attack_type is not None):
        simul_name += "-" + attack_type
        if (attack_type == AttackEnum.RANDOM_FULL_MODEL):
            simul_name += "-m" + str(attack_mean)
        if (attack_type == AttackEnum.GRADIENT_SCALING or attack_type == AttackEnum.RANDOM_FULL_MODEL or attack_type == AttackEnum.RANDOM_GRADIENT):
            simul_name += "-s" + str(attack_scale)
        if (model_handler_type == ModelHandlerEnum.PEGASOS):
            simul_name += "-nb" + str(attack_pegasos_nb)

        if (generation_type != GenerationTypeEnum.NORMAL):
            simul_name += "-" + generation_type

        folder_name = result_folder + "/" + simul_name

        if (attack_own_data):
            simul_name += "-" + str(attack_proportion*100) + "%"
        else:
            simul_name += "-+" + str(attack_proportion*100) + "%"

    makedirs(folder_name, exist_ok=True)

    with open(folder_name + "/" + simul_name + ".args.pickle", "wb") as file:
        file.write(local_args)

    with open(folder_name + "/" + simul_name + ".pickle", "wb") as file:
        pickle.dump([[ev for _, ev in report.get_evaluation(False)]], file)

    if (topology_matrix is not None):
        with open(folder_name + "/" + simul_name + ".net.pickle", "wb") as file:
            pickle.dump(topology_matrix, file)

    plot_evaluation([[ev for _, ev in report.get_evaluation(False)]],
                    "Overall test results", folder_name + "/" + simul_name + ".png")


@app.command()
def verify_args(model_handler_type: ModelHandlerEnum = ModelHandlerEnum.TORCH_MODEL,
                create_model_mode: CreateModelModeEnum = CreateModelModeEnum.MERGE_UPDATE,
                torch_net: Optional[TorchNetEnum] = None,
                torch_activation: Optional[TorchActivationEnum] = None,
                torch_criterion: Optional[TorchCriterionEnum] = None,
                torch_local_epochs: int = 1,
                torch_batch_size: int = 32,
                torch_hidden_layer_dim: List[int] = [],
                learning_rate: float = 0.01,
                weight_decay: float = 0,
                sample_size: Optional[float] = None,
                n_parts: Optional[int] = None,
                kmeans_k: Optional[int] = None,
                kmeans_alpha: float = 0.1,
                kmeans_matching: KMeansMatchingEnum = KMeansMatchingEnum.NAIVE,
                dataset_name: DataSetEnum = DataSetEnum.SPAMBASE,
                dataset_path: Optional[Path] = None,
                dataset_normalize: bool = True,
                dataset_widerange: Optional[bool] = None,
                dataset_test_proportion: float = 0.1,
                dataset_use_test: bool = True,
                data_handler_type: DataHandlerEnum = DataHandlerEnum.CLASSIFICATION,
                topology_type: TopologyEnum = TopologyEnum.PLAIN,
                barabasi_m: Optional[int] = None,
                erdos_renyi_prob: Optional[float] = None,
                clique_size: Optional[List[int]] = None,
                n_nodes: Optional[int] = None,
                n_rounds: int = 150,
                node_type: NodeEnum = NodeEnum.GOSSIP,
                sync_nodes: bool = False,
                round_len: int = 100,
                pens_n_sample: int = 10,
                pens_m_top: int = 2,
                pens_step1_rounds: int = 100,
                generation_type: GenerationTypeEnum = GenerationTypeEnum.NORMAL,
                simulator_type: SimulatorEnum = SimulatorEnum.GOSSIP,
                delta: int = 100,
                anti_entropy_protocol: AntiEntropyProtocolEnum = AntiEntropyProtocolEnum.PUSH,
                drop_prob: float = 0.,
                online_prob: float = 1.,
                delay_mode: Optional[DelayEnum] = None,
                delay_constant_value: Optional[int] = None,
                delay_min: Optional[int] = None,
                delay_max: Optional[int] = None,
                delay_factor: Optional[float] = None,
                sampling_eval: float = 0.,
                attack_type: Optional[AttackEnum] = None,
                attack_proportion: Optional[float] = None,
                attack_own_data: bool = True,
                attack_scale: Optional[float] = None,
                attack_mean: Optional[float] = None,
                attack_pegasos_nb: int = 1,
                seed: int = 42,
                on_device: bool = True,
                result_folder: str = "results",
                simul_name: Optional[str] = None):
    if model_handler_type in (ModelHandlerEnum.TORCH_MODEL, ModelHandlerEnum.SAMPLING_TMH, ModelHandlerEnum.PARTITIONED_TMH):
        assert torch_net is not None, "torch_net should be set with that model handler"
        assert torch_criterion is not None, "torch_criterion should be set with that model handler"
        if torch_net in (TorchNetEnum.MLP, TorchNetEnum.MLP_SOFTMAX):
            assert torch_activation is not None, "torch_activation should be set with that torch_net"
            assert len(
                torch_hidden_layer_dim) > 0, "torch_hidden_layer_dim should be set with that torch_net"
    elif torch_net is not None or torch_criterion is not None or torch_activation is not None or len(torch_hidden_layer_dim) != 0:
        print("WARNING : Some torch_* are set but they will be useless.")

    if model_handler_type == ModelHandlerEnum.KMEANS:
        assert on_device == False, "KMEANS does not support GPU for now"

    if model_handler_type == ModelHandlerEnum.SAMPLING_TMH:
        assert sample_size is not None, "sample_size should be set with that model handler"
    elif sample_size is not None:
        print("WARNING : sample_size is set but it will be useless with that model handler")

    if model_handler_type == ModelHandlerEnum.PARTITIONED_TMH:
        assert n_parts is not None, "n_parts should be set with that model handler"
    elif n_parts is not None:
        print("WARNING : n_parts is set but it will be useless with that model handler")

    if model_handler_type == ModelHandlerEnum.KMEANS:
        assert kmeans_k is not None, "kmeans_k should be set with that model handler"
    elif kmeans_k is not None:
        print("WARNING : kmeans_k has been set but it is useless")

    if dataset_name == DataSetEnum.PATH:
        assert dataset_path is not None, "dataset_path should be set with that dataset_name"
    elif dataset_path is not None:
        print("WARNING : dataset_path has been set but it won't be taken into account with that dataset_name")

    if model_handler_type in (ModelHandlerEnum.ADALINE, ModelHandlerEnum.PEGASOS) and dataset_widerange == False:
        print(
            "WARNING : Adaline and Pegasos should be used with wide range data [-1, 1]")

    if topology_type == TopologyEnum.BARABASI_ALBERT:
        assert barabasi_m is not None, "barabasi_m should be set with that topology_type"
    elif barabasi_m is not None:
        print("WARNING : barabasi_m is set but it wil be useless with that topology_type")

    if topology_type == TopologyEnum.ERDOS_RENYI:
        assert erdos_renyi_prob is not None, "erdos_renyi_prob should be set with that topology_type"
    elif erdos_renyi_prob is not None:
        print("WARNING : erdos_renyi_prob is set but it wil be useless with that topology_type")

    if topology_type == TopologyEnum.CLIQUES:
        assert clique_size is not None and len(
            clique_size) > 0, "clique_size should be set with that topology_type"
    elif clique_size is not None and len(clique_size) > 0:
        print("WARNING : clique_size is set but it wil be useless with that topology_type")

    if delay_mode in (DelayEnum.CONSTANT, DelayEnum.LINEAR):
        assert delay_constant_value is not None, "delay_constant_value should be set with that delay_mode"
    elif delay_constant_value is not None:
        print("WARNING : delay_constant_value is set but it wil be useless with that delay_mode")

    if delay_mode == DelayEnum.UNIFORM:
        assert delay_min is not None and delay_max is not None, "delay_min and delay_max should be set with that delay_mode"
    elif delay_min is not None and delay_max is not None:
        print("WARNING : delay_min and/or delay_max are set but they wil be useless with that delay_mode")

    if delay_mode == DelayEnum.LINEAR:
        assert delay_factor is not None, "delay_factor should be set with that delay_mode"
    elif delay_factor is not None:
        print("WARNING : delay_factor is set but it wil be useless with that delay_mode")

    if attack_type is not None:
        assert attack_proportion is not None, "attack_proportion should be set with that attack_type"
    elif attack_proportion is not None:
        print("WARNING : attack_proportion is set but it will be useless with that attack_type")

    if attack_type in (AttackEnum.GRADIENT_SCALING, AttackEnum.RANDOM_FULL_MODEL, AttackEnum.RANDOM_GRADIENT):
        assert attack_scale is not None, "attack_scale should be set with that attack_type"
    elif attack_scale is not None:
        print("WARNING : attack_scale is set but it will be useless with that attack_type")

    if attack_own_data != True:
        assert attack_type not in (
            AttackEnum.BACKGRADIENT, AttackEnum.GRADIENT_SCALING), "Malicious nodes must own data for this attack_type."

    if attack_type == AttackEnum.RANDOM_FULL_MODEL:
        assert attack_mean is not None, "attack_mean should be set with that attack_type"
    elif attack_mean is not None:
        print("WARNING : attack_mean is set but it will be useless with that attack_type")


def save_args(**kwargs):
    return pickle.dumps(kwargs)


@app.command()
def run_file(path: Path):
    with open(path, "rb") as file:
        run(**pickle.load(file))


class PresetEnum(str, Enum):
    BERTA_2014 = "berta2014"
    ORMANDI_2013 = "ormandi2013"
    GIARETTA_2019 = "giaretta2019"

    def __str__(self):
        return self.value


@app.command()
def preset(preset_name: PresetEnum = typer.Argument(..., help="Name of the preset", case_sensitive=False),
           attack_type: Optional[AttackEnum] = typer.Option(
    None, help="Type of malicious nodes. Leave it if you don't want to try any byzantine attack and you just want a simulation", case_sensitive=False),
    attack_proportion: Optional[float] = typer.Option(
    None, help="Proportion of malicious nodes", case_sensitive=False),
    attack_own_data: bool = typer.Option(
    True, help="Wether malicious nodes own data or they are added to the simulation", case_sensitive=False),
    attack_scale: Optional[float] = typer.Option(
    None, help="Scale parameters for attacks (depends on attack type)", case_sensitive=False),
    attack_mean: Optional[float] = typer.Option(
    None, help="Mean parameters for attacks (depends on attack type)", case_sensitive=False),
    attack_pegasos_nb: int = typer.Option(
    1, help="Value added to n_update whith a pegasos attack", case_sensitive=False),
):
    if preset_name == PresetEnum.BERTA_2014:
        run_impl(model_handler_type=ModelHandlerEnum.KMEANS, kmeans_k=2, kmeans_alpha=0.1, kmeans_matching=KMeansMatchingEnum.HUNGARIAN, dataset_name=DataSetEnum.SPAMBASE, data_handler_type=DataHandlerEnum.CLUSTERING, topology_type=TopologyEnum.PLAIN,
                 n_rounds=500, sync_nodes=True, round_len=1000, delta=1000, drop_prob=.1, sampling_eval=0.01, simul_name="berta2014", on_device=False, attack_type=attack_type, attack_proportion=attack_proportion, attack_own_data=attack_own_data, attack_mean=attack_mean, attack_scale=attack_scale, attack_pegasos_nb=attack_pegasos_nb)
    elif preset_name == PresetEnum.ORMANDI_2013:
        run_impl(model_handler_type=ModelHandlerEnum.PEGASOS, learning_rate=0.01, dataset_name=DataSetEnum.SPAMBASE, dataset_widerange=True, dataset_test_proportion=.1,
                 n_rounds=100, round_len=100, delta=100, drop_prob=.1, online_prob=.2, delay_mode=DelayEnum.UNIFORM, delay_min=0, delay_max=10, sampling_eval=0.1, seed=42, on_device=True, simul_name="ormandi2013", attack_type=attack_type, attack_proportion=attack_proportion, attack_own_data=attack_own_data, attack_mean=attack_mean, attack_scale=attack_scale, attack_pegasos_nb=attack_pegasos_nb)
    elif preset_name == PresetEnum.GIARETTA_2019:
        run_impl(model_handler_type=ModelHandlerEnum.PEGASOS, learning_rate=0.01, dataset_name=DataSetEnum.SPAMBASE, dataset_widerange=True, dataset_test_proportion=.1,
                 n_rounds=100, round_len=100, delta=100, topology_type=TopologyEnum.BARABASI_ALBERT, barabasi_m=10, sampling_eval=0.1, seed=42, on_device=True, simul_name="giaretta2019", attack_type=attack_type, attack_proportion=attack_proportion, attack_own_data=attack_own_data, attack_mean=attack_mean, attack_scale=attack_scale, attack_pegasos_nb=attack_pegasos_nb)
    else:
        raise RuntimeError("Wrong name of preset")


if __name__ == "__main__":
    app()
