import numpy as np
import networkx as nx
import torch
import random
import os
import matplotlib.pyplot as plt
from time import time
from typing import Optional, Dict, Any
from problems import get_synthetic_problem
from search.baselines import DFS_BFS_Search, Local_Search
from search.graphbo import GraphBO_Search
from search.utils import prune_baseline, generate_neighbors, get_context_graph, eigendecompose_laplacian
from search.trust_region import update_state, restart, restart1

def run_search(
        label: str,
        seed: int,
        problem_name: str,
        save_path: str,
        iterations: int = 100,
        batch_size: int = 1,
        n_initial_points: Optional[int] = None,
        acqf_optimizer: str = "enumerate",
        max_radius: int = 10,
        Q: int = 100,
        k: int = 2,
        acqf_kwargs: Optional[dict] = None,
        acqf_optim_kwargs: Optional[dict] = None,
        model_optim_kwargs: Optional[Dict[str, Any]] = None,
        trust_region_kwargs: Optional[Dict[str, Any]] = None,
        problem_kwargs: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = torch.float,
        device: str = "cpu",
        save_frequency: int = 1,
        animation: bool = False,
        animation_interval: int = 20,
        order=None,):
    print(f"Using {label} method...")
    graph_kernels = ["polynomial","polynomial_suminverse","diffusion","diffusion_ard"]
    trust_region_kwargs = trust_region_kwargs or {}
    problem_kwargs = problem_kwargs or {}

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_path = os.path.join(save_path, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tkwargs = {"dtype": dtype, "device": device}
    acqf_optim_kwargs = acqf_optim_kwargs or {}
    acqf_kwargs = acqf_kwargs or {}
    model_optim_kwargs = model_optim_kwargs or {}
    
    # ------------- Initialise the Problem --------------
    base_function = get_synthetic_problem(problem_name, seed=seed, 
                                          problem_kwargs=problem_kwargs)
    
    ground_truth = base_function.ground_truth.cpu()

    # Initialise context subgraph size Q
    Q = min(Q, len(base_function.context_graph)) if Q else len(base_function.context_graph) 
    n_initial_points = n_initial_points or 20 # Initial queried points
    use_trust_region = label in graph_kernels
    n_restart = 0
    
    candidates, trust_region_state = restart1( # Initial queried locations
        base_graph=base_function.context_graph,
        n_init=n_initial_points,
        seed=seed,
        k=3,
        batch_size=batch_size,
        init_context_graph_size=Q, # Initial subgraph size
        n_restart=n_restart,
        iterations=iterations,
        use_trust_region=use_trust_region,
        options=trust_region_kwargs,)
    __import__("pdb").set_trace() 
    X_queried = candidates.reshape(-1, 1).to(**tkwargs) # X_queried/Y_queried is the set of queried nodes/values.
    #####
    Y_queried = base_function(X_queried).to(**tkwargs) # X_queried is the global indices (i.e. on the original graph).
    X_train = X_queried.clone() # X_train, Y_train are the training sets, i.e. nodes inside subgraph at each iteration.
    Y_train = Y_queried.clone() # They will be updated at each iteration by "prune_baseline".
    # Note X_train/Y_train are useless for the baselines, they are just the same as X_queried/Y_queried.

    # Set some counters to keep track of things.
    start_time = time()
    existing_iterations = 0
    wall_time = torch.zeros(iterations, dtype=dtype)
    best_obj = Y_queried.max().view(-1)[0].cpu()
    best_loc = X_queried[Y_queried.argmax().view(-1)[0]].cpu()
    if acqf_optimizer is None:
        acqf_optimizer = "enumerate" if base_function.problem_size <= 1000 else "local_search"
    
    # ----------------- Initialise the Models ----------------
    if label in graph_kernels: # initialise BO models
        context_graph = get_context_graph(base_function.context_graph,
                                          best_loc.item(),
                                          nnodes=Q,) # get the initial context graph
        
        # Only include nodes inside subgraph for training
        X_train, Y_train = prune_baseline(X_queried, Y_queried, torch.tensor(list(context_graph.nodes)).to(X_queried))
        
        # index lookup:local index: global index
        inverse_map_dict = dict(zip(
            list(range(context_graph.number_of_nodes())),
            list(context_graph.nodes) ))
        # global index -> local index
        map_dict = {v: k for k, v in inverse_map_dict.items()}
        # functions to create the indices in terms of the global graph and the local graph
        def index_remapper(x): return torch.tensor(
            [map_dict[int(i)] for i in x]).to(x).reshape(-1, x.shape[-1])
        def inverse_index_remapper(x): return torch.tensor(
            [inverse_map_dict[int(i)] for i in x]).to(x).reshape(-1, x.shape[-1])
        
        # Set up the configurations for surrogate models with different kernels
        model_configurations = {"covar_type":label, "order":None, "ard": True}
        if label == "diffusion_ard": # both diffusion_ard and diffusion share the same kernel 
            model_configurations["covar_type"] = "diffusion" # the differences are "ard" and "order"
            model_configurations["order"] = len(context_graph.nodes) ## Change to order size context graph
        elif label == "diffusion":
            model_configurations["ard"] = False # Diffusion kernel without ARD
    
    elif label in ["dfs", "bfs", "local_search"]: # Initialise baselines 
        context_graph = base_function.context_graph # Use the entire graph as context subgraph
        if label in ["dfs", "bfs"]: # Initialise the stacks for DFS/BFS
            list_stacks = []
            for i in range(batch_size):
                neighbors_current = generate_neighbors(int(X_train[-i]), base_function.context_graph,
                                                       X_avoid=X_queried)
                list_stacks.append(list(neighbors_current.numpy().flatten()))

    # If the subgraph doesn't change from last iter, then no need for eigendecomposition
    cached_eigenbasis = None
    use_cached_eigenbasis = True
    i=0

    # ---------------- Search Starts ----------------
    while len(X_queried) < iterations:
        # --------------- Query Point Selection ------------------ 
        if label == "random":
            candidates = torch.from_numpy(
                    np.random.RandomState(seed).choice(
                        base_function.problem_size,
                        size=iterations)).reshape(-1, 1).to(**tkwargs)
        elif label in ["bfs", "dfs"]:
            candidates, list_stacks = DFS_BFS_Search(label, X_queried, list_stacks, 
                                context_graph, n_initial_points, seed, batch_size, 
                                trust_region_kwargs)
        elif label == "local_search":
            candidates = Local_Search(label,X_queried, best_loc, context_graph,
                                      n_initial_points, seed, batch_size,
                                      trust_region_kwargs)
        # Use Graph BO methods for query point selection 
        elif label in graph_kernels:
            X_train, Y_train, candidates, trust_region_state,\
            n_restart, cached_eigenbasis = GraphBO_Search(label, X_queried, X_train, Y_train, n_restart, 
                                                          base_function, context_graph, n_initial_points, 
                                                          seed, batch_size, Q, model_configurations, 
                                                          index_remapper, inverse_index_remapper, 
                                                          cached_eigenbasis, use_cached_eigenbasis, 
                                                          trust_region_state, trust_region_kwargs, 
                                                          acqf_kwargs, acqf_optim_kwargs,
                                                          model_optim_kwargs)
        
        # --------------- Query Point Evaluation -------------------
        if candidates is None:
            continue
        #####
        new_y = base_function(candidates) # Query the selected location
        X_queried = torch.cat([X_queried, candidates], dim=0) # append the new queried point to the queried set
        Y_queried = torch.cat([Y_queried, new_y], dim=0)
        X_train = torch.cat([X_train, candidates], dim=0)
        Y_train = torch.cat([Y_train, new_y], dim=0) # append the new queried point to the training set
        new_best_obj = Y_train.max().view(-1)[0].cpu()
        
        # update the trust region state, if applicable
        if use_trust_region:
            trust_region_state = update_state(state=trust_region_state, Y_next=new_y)
        
        # Recompute the subgraph at new centre if best location changes
        if (label in graph_kernels or label == "local_search") \
                and (new_best_obj != best_obj or context_graph is None):
            best_idx = Y_train.argmax().cpu()
            print(f'Context subgraph centre changes from {best_loc} to {X_train[best_idx]}!')
            best_loc = X_train[best_idx]
            if label in graph_kernels:
                context_graph = get_context_graph(base_function.context_graph,
                                                  best_loc.item(),
                                                  nnodes=Q)
                X_train, Y_train = prune_baseline(X_queried, Y_queried, torch.tensor(list(context_graph.nodes)).to(X_queried))
                # the context graph changed -- need to re-compute the eigenbasis for the next BO iteration.
                inverse_map_dict = dict(zip(list(range(context_graph.number_of_nodes())),
                                            list(context_graph.nodes)))
                map_dict = {v: k for k, v in inverse_map_dict.items()}
                use_cached_eigenbasis = False
        else: # just use the subgraph from last iteration
            use_cached_eigenbasis = True
        best_obj = new_best_obj
        
        wall_time[i] = time() - start_time
        if new_y.shape[0] == 1:
            context_nodes = context_graph.number_of_nodes() if "ego_network" in label else None
            print(f'Seed{seed} iter:{i}, Candidate:{str(candidates.long().item()).zfill(4)}, ' 
                  f'current obj:{new_y.item():.4f}, best obj:{Y_queried.max().squeeze().item():.4f}, '
                  f'truth:{ground_truth:.4f}, #Queried:{len(Y_queried)}, #Training:{len(Y_train)}')
        i+=1 # counting the iterations, note usually it will be smaller than len(X_queried)

    # ------------------ Save the Final Output --------------------
    if hasattr(base_function, "ground_truth"):
        regret = base_function.ground_truth.cpu() - Y_queried.cpu()
    else:
        regret = None
    output_dict = {"label": label, "X": X_queried.cpu(), "Y": Y_queried.cpu(), "wall_time": wall_time,
                   "best_obj": best_obj, "regret": regret,}
    with open(os.path.join(save_path, f"{str(seed).zfill(4)}_{label}.pt"), "wb") as fp:
        torch.save(output_dict, fp)
