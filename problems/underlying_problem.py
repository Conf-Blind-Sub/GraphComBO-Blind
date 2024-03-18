import networkx as nx
import numpy as np
import random
import torch
import pickle
import time
import os
import future.utils
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import osmnx as ox
from math import sqrt, comb
from functools import partial
from torch_geometric.datasets import Coauthor, FacebookPagePage, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx, from_networkx 
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from typing import Optional, Any, Union, Tuple, Callable, Dict
from search.utils import eigendecompose_laplacian, graph_relabel
from utils.config_utils import setup
from problems.GNN.GNN_attack import GIN, generate_prediction
from problems.SIR import SIR_MC, One_SIR_Run
from problems.IC import IC_MC

# Formulate the underlying problem as Class object
class Problem:
    def __init__(self,
                 underlying_graph: nx.Graph,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 log: bool = False):
        self.noise_std = noise_std
        self.negate = negate
        self.log = log
        self.underlying_graph = underlying_graph
        self.problem_size = None
    def get_context_adj_mat(self) -> torch.Tensor:
        A = nx.to_numpy_array(self.underlying_graph)
        A = torch.from_numpy(A)
        return A
    @property
    def is_moo(self) -> bool:
        raise NotImplementedError
    def evaluate_true(self, X):
        raise NotImplementedError
    def __call__(self, X: torch.Tensor, noise: bool = False):
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X).to(dtype=torch.float, device=X.device)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        if self.log:
            f = torch.log(f) # f = torch.log(f + 1e-4)
        f += 1e-6 * torch.randn_like(f)
        return f if batch else f.squeeze(0)

class SyntheticProblem(Problem):
    is_moo = False
    def __init__(self,
                 underlying_graph,
                 obj_func: Callable,
                 ground_truth: torch.Tensor = None,
                 problem_size: Optional[int] = None,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 log: bool = False,
                 **kwargs,
                 ):
        super().__init__(underlying_graph, noise_std=noise_std, negate=negate, log=log)
        self.obj_func = obj_func
        self.problem_size = problem_size
        self.ground_truth = ground_truth

    @torch.no_grad()
    def evaluate_true(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().long()
        return self.obj_func(X)

# Define the underlying problem details
def get_synthetic_problem(
        label: str,
        seed: int = 0,
        problem_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SyntheticProblem":
    problem_kwargs = problem_kwargs or {}
    n = problem_kwargs.get("n", 5000)
    k = problem_kwargs.get("k", 2)
    graph_type = problem_kwargs.get("graph_type", "ba")
    
    # ----------------- Underlying Graphs --------------------
    if graph_type == "ba":
        m = problem_kwargs.get("m", 1)
        g = nx.generators.random_graphs.barabasi_albert_graph(
            seed=seed, n=n, m=m)
        print(f"Using BA network (n={n} m={m}) with {problem_kwargs['underlying_function']} defined on {k} combinations of nodes")
    elif graph_type == "ws":
        wsk = problem_kwargs.get("wsk", 10)
        p = problem_kwargs.get("p", 0.1)
        g = nx.generators.random_graphs.watts_strogatz_graph(
            n=n, k=wsk, p=p, seed=seed)
        print(f"Using WS network (n={n} k={wsk} p={p}) with {problem_kwargs['underlying_function']} defined on {k} combinations of nodes")
    elif graph_type == "grid":
        print(f"Using grid network (n={n}) with {problem_kwargs['underlying_function']} defined on {k} combinations of nodes")
        n, m = int(sqrt(n)), int(sqrt(n))
        g = nx.generators.grid_2d_graph(n, m)
        mapping = {}
        for i in range(n):
            for j in range(m):
                mapping[(i, j)] = i * m + j
        g = nx.relabel_nodes(g, mapping)
    elif graph_type in ["contact_network_day1", "contact_network_day2"]:
        # Two real-world contact networks in a French primary school: day1 and day2
        current_dir = os.path.dirname(os.path.abspath(__file__))
        g,_,_ = graph_relabel(nx.read_gexf(f'{current_dir}/primary_school_{graph_type}.gexf_'))
        print(f"Using epidemic {graph_type} (n={g.number_of_nodes()}) to select {k} nodes for protection "
              f"that maximizes the expected time to reach {problem_kwargs['infection_percentage_threshold']} population infection.")
    elif graph_type in ["CS"]:
        dataset = Coauthor(root='./problems/Coauthor', name='CS')
        g = to_networkx(dataset[0])
        print(f"Using a coauthor network: {graph_type} (n={g.number_of_nodes()}) to select {k} nodes for protection.")
    elif graph_type in ["Facebook"]:
        dataset = FacebookPagePage(root='./problems/Facebook')
        g = to_networkx(dataset[0])
        print(f"Using {graph_type} Page-Page network (n={g.number_of_nodes()}) to select {k} nodes for protection.")
    elif graph_type in ["Road"]:
        road_network = ox.graph_from_place('Manhattan, New York City, NY, USA', network_type='drive')
        #Road_Network = ox.graph_from_place('City of Westminster, London, England, UK', network_type='drive')
        graph = nx.Graph(road_network.to_undirected()) # Here we make it undirected and without multi-edges.
        graph_line = nx.line_graph(graph) # Convert it to line graph such that roads become nodes.
        g, _, inverse_mapping = graph_relabel(graph_line)
        print(f"Using {graph_type} network (|V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}) "
              f"(converted to a line graph in search), where we wish to delete {k} edges (i.e. nodes on the "
              f"line graph) that minimises the transitivity (a.k.a. global clustering coefficient).")
    elif graph_type in ["ENZYMES","DD"]:
        idx_graph_to_attack = problem_kwargs.get("graph_index", 2)
        dataset = TUDataset(root='./problems/TUDataset', name=graph_type)
        graph_pyg = dataset[idx_graph_to_attack]
        graph = to_networkx(graph_pyg).to_undirected()
        graph_line = nx.line_graph(graph) # Convert it to line graph such that edges represent nodes.
        g, _, inverse_mapping = graph_relabel(graph_line)
        print(f"Using {graph_type} network (|V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}) "
              f"(converted to a line graph in search), where we wish to delete {k} edges (i.e. nodes on the "
              f"line graph) that maximises the JS divergence of the GNN prediction at softmax.")
    else:
        raise ValueError(f"Unknown graph type {graph_type}")

    # ------------------ Underlying Functions -------------------
    total_comb = comb(g.number_of_nodes(), k)
    
    if label in ["synthetic"]:  # alias for backward compatibilitys
        feature_name = problem_kwargs.get("underlying_function", "eigenvector_centrality")
        # Here because of experimental purposes we can first compute the underlying 
        # functions and get the ground truth (global maxima of average centrality)
        if feature_name == "betweenness_centrality":
            feature_dict = nx.betweenness_centrality(g)
            feature = torch.tensor(list(feature_dict.values()))
        elif feature_name == "eigenvector_centrality":
            feature_dict = nx.eigenvector_centrality(g, max_iter=1000)
            feature = torch.tensor(list(feature_dict.values()))
        elif feature_name == "eigenvector":
            _, laplacian_eigenvecs = eigendecompose_laplacian(
                g, normalized_laplacian=True, normalized_eigenvalues=True,)
            feature = laplacian_eigenvecs[:, problem_kwargs.get("j", 2)]
        elif feature_name == "ackley":
            n = problem_kwargs.get("n", 5000)
            noise = problem_kwargs.get("noise", 0.)
            n, m = int(sqrt(n)), int(sqrt(n))
            feature_dict = dict.fromkeys(range(n), 0)
            def test_fun(x, y):
                return (-1)*(-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20)
            for i in range(n):
                for j in range(m):
                    feature_dict[i * m + j] = (test_fun(1 - (2 / n) * i, -1 + (2 / m) * j)) + np.random.normal(loc = 0., scale=noise)
            feature = torch.tensor(list(feature_dict.values()))
        feature_sorted, idx = feature.sort()
        print("computing ground truth for synthetic problems ...........")
        ground_truth = feature_sorted[-k:].mean()
        # query function 
        def obj_func(combo_node): return compute_synthetic_node_features(combo_node, g, feature_name=feature_name, feature=feature)
        return SyntheticProblem(g, obj_func, ground_truth, problem_size=total_comb, **problem_kwargs)
    
    elif label in ["epidemic"]:
        print(f"Using {problem_kwargs['SIR_n_iterations']} iterations in each SIR simulation and "
              f"repeat {problem_kwargs['SIR_n_samples']} times to estimate the expected time of infecting the target proportion in population.")
        feature_name = problem_kwargs.get("underlying_function", "infection_time")
        def obj_func(combo_node): return compute_synthetic_node_features(combo_node, g, feature_name=feature_name,**problem_kwargs)
        return SyntheticProblem(g, obj_func, problem_size=total_comb, **problem_kwargs)
    elif label in ["influence_maximisation"]:
        print(f"Using p={problem_kwargs['IC_p']} in each IC simulation and "
              f"repeat {problem_kwargs['IC_n_samples']} times to estimate the expected number of influence.")
        feature_name = problem_kwargs.get("underlying_function", "independent_cascading")
        def obj_func(combo_node): return compute_synthetic_node_features(combo_node, g, feature_name=feature_name,**problem_kwargs)
        return SyntheticProblem(g, obj_func, problem_size=total_comb, **problem_kwargs)
    elif label in ["resilience"]:
        feature_name = problem_kwargs.get("underlying_function", "transitivity")
        def obj_func(combo_node):
            return compute_synthetic_node_features(combo_node, graph, feature_name=feature_name, 
                                                   inverse_mapping=inverse_mapping, **problem_kwargs)
        return SyntheticProblem(g, obj_func, problem_size=total_comb, **problem_kwargs)
    elif label in ["gnn_attack"]:
        feature_name = problem_kwargs.get("underlying_function", "JS")
        gnn_config = setup('./problems/GNN/config.yaml') # load the GNN configs
        pred_base = generate_prediction(graph_pyg, dataset.num_classes, gnn_config)# prediction on the original graph
        def obj_func(combo_node):
            return compute_synthetic_node_features(combo_node, graph_pyg, feature_name=feature_name, 
                                                   inverse_mapping=inverse_mapping, gnn_args=gnn_config,
                                                   num_classes=dataset.num_classes, pred_base=pred_base)
        return SyntheticProblem(g, obj_func, problem_size=total_comb, **problem_kwargs)
    else:
        raise NotImplementedError(f"Problem {label} is not implemented")

# Compute the underying functions
def compute_synthetic_node_features(
        combo_node: torch.Tensor,
        input_graph: nx.Graph,
        feature_name: str = "betweenness_centrality",
        model: ep.SIRModel = None,
        forward_dict: dict = None,
        feature=None,
        inverse_mapping: dict = None,
        **problem_kwargs):
    nnodes = len(input_graph)
    if feature is not None: # A synthetic setting by averaging pre-computed node-features over nodes.
        ret = list(map(lambda x: feature[combo_node[x]].mean(),
                                    range(combo_node.shape[0])))
    else: # This is for real-world scenarios when we don't have pre-computed synthetic features 
        start_time = time.time()
        ret = [] # an empty list to store evaluation results
        if feature_name == "infection_time":
            cfg = mc.Configuration()
            cfg.add_model_parameter('beta', 0.001)
            cfg.add_model_parameter('gamma', 0.01)
            cfg.add_model_parameter("fraction_infected", 0.1)
            for ComboNode in combo_node:
                partial_function = partial(One_SIR_Run, input_graph, cfg, ComboNode.tolist(),
                                           problem_kwargs.get("SIR_n_iterations",500))
                function_value = SIR_MC(T=problem_kwargs.get("SIR_n_iterations",500),
                                        N=input_graph.number_of_nodes(),
                                        n_samples=problem_kwargs.get("SIR_n_samples", 100),
                                        threshold=problem_kwargs.get("infection_percentage_threshold",0.5),
                                        parallel_function=partial_function)
                ret.append(function_value)
        elif feature_name == "independent_cascading":
            for ComboNode in combo_node:
                function_value = IC_MC(input_graph, 
                                       ComboNode.tolist(), 
                                       p=problem_kwargs.get("IC_p",0.05), 
                                       mc=problem_kwargs.get("IC_n_samples",1000))
                ret.append(function_value)
        elif feature_name == "transitivity": # Note the underlying graph is a line graph
            for ComboNode in combo_node:
                edges_to_remove = [inverse_mapping[i.item()] for i in ComboNode] # map combonode back to edge-tuple labels
                G = input_graph.copy()
                G.remove_edges_from(edges_to_remove)
                ret.append(-round(nx.transitivity(G),4))
        elif feature_name in ["JS", 'WD']: # Note the underlying graph is a line graph
            for ComboNode in combo_node:
                edges_to_remove = torch.tensor([inverse_mapping[i.item()] for i in ComboNode]) # map combonode back to edge-tuple labels
                edges_to_remove_ready = torch.vstack([edges_to_remove, # stack the flipped edges since pyg store edges in two directions
                                                      torch.flip(edges_to_remove,[-1])]) # e.g. [0, 2] and [2, 0] are both in edge_index
                perturbed_graph = input_graph.clone()
                X = perturbed_graph.edge_index.T
                X1 = X[:,None,:] # broadcasting 
                mask = (X1 == edges_to_remove_ready).all(dim=-1).any(dim=-1) # create a mask for the removed edges
                perturbed_graph.edge_index = X[~mask].T # assign the perturbed edge index back to graph
                pred_attack = generate_prediction(perturbed_graph, problem_kwargs['num_classes'], 
                                                  problem_kwargs['gnn_args'])
                if feature_name == 'JS':
                    dis = distance.jensenshannon(pred_attack, problem_kwargs['pred_base'])
                elif feature_name == 'WD':
                    dis = wasserstein_distance(pred_attack, problem_kwargs['pred_base'])
                ret.append(round(dis,4))
        else:
            raise ValueError(f"Unknown feature name {feature_name}")

        end_time = time.time()
        if feature_name in ["independent_cascading", "infection_time"]:
            print(f"time cost for evaluation: {(end_time - start_time):.4f}s")

    return torch.tensor(ret).reshape(-1,1)
