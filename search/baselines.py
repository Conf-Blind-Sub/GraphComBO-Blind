import numpy as np
import networkx as nx
import torch
import random
from search.utils import ComboNeighbors_Generator
from search.trust_region import restart

# -------------------- Baselines -------------------
def DFS_BFS_Search(Problem,
                   X_queried,
                   label,
                   list_stacks,
                   seed,
                   k,
                   anchor,
                   n_initial_points, 
                   batch_size,
                   n_restart,
                   iterations,
                   ComboGraph,
                   ):
    visited = set(tuple(row.int().tolist()) for row in X_queried)
    flag = 1
    for stack in list_stacks:
        flag *= len(stack)
    if flag:
        candidates = []
        for i_stack, stack in enumerate(list_stacks):
            element = stack.pop()
            if element not in visited:
                neighbors_element = ComboNeighbors_Generator(Problem.underlying_graph, 
                                                             torch.tensor(element), 
                                                             X_avoid=X_queried)
                neighbors_element_tuple = [tuple(x) for x in neighbors_element.tolist()]
                random.shuffle(neighbors_element_tuple) # make the algorithm random from different runs
                # Then do DFS or BFS
                if label == "dfs":
                    stack = stack + neighbors_element_tuple
                elif label == "bfs":
                    stack =  neighbors_element_tuple + stack
                list_stacks[i_stack] = stack
                
                # Now we add these new graph structure to ComboGraph for recording purposes
                if len(neighbors_element):
                    center_combonode_to_stack = np.repeat(np.array(element).reshape(1,-1),len(neighbors_element),axis=0)
                    comboedges_array_to_add = np.stack((neighbors_element, center_combonode_to_stack),axis=1).astype(int)
                    comboedges_to_add = list(tuple(map(tuple, i)) for i in comboedges_array_to_add) # Convert np.array to list of tuples
                    ComboGraph.add_edges_from(comboedges_to_add) # record the explored structure

            candidates.append(list(element))
        candidates = torch.tensor(candidates)
    else: #Restard if stuck
        print(f"=========== Restart Triggered ==============")
        candidates, trust_region_state = restart( # Initial queried locations
            base_graph=Problem.underlying_graph,
            n_init=n_initial_points,
            seed=seed,
            k=k,
            batch_size=batch_size,
            X_avoid=X_queried,
            anchor=None,
            n_restart=n_restart,
            iterations=iterations,
            use_trust_region=False,)
    return candidates, list_stacks, ComboGraph

def Local_Search(Problem,
                 X_queried,
                 X_train,
                 Y_train,
                 best_loc,
                 seed,
                 k,
                 anchor,
                 n_initial_points,
                 batch_size,
                 n_restart,
                 iterations,
                 ComboGraph,):
    neighbors_of_best = ComboNeighbors_Generator(Problem.underlying_graph, best_loc, X_avoid=X_queried)
    # when we cannot find a valid point for the local search, we have reached a local minimum.
    # randomly spawn a new starting point
    if not len(neighbors_of_best): # restart if there is no neighbor around best_loc
        print(f"=========== Restart Triggered ==============")
        candidates, trust_region_state = restart( # Initial queried locations
            base_graph=Problem.underlying_graph,
            n_init=n_initial_points,
            seed=seed,
            k=k,
            batch_size=batch_size,
            X_avoid=X_queried,
            anchor=None,
            n_restart=n_restart,
            iterations=iterations,
            use_trust_region=False,)
        X_train = torch.zeros(0, X_train.shape[1]).to(X_train)
        Y_train = torch.zeros(0, 1).to(Y_train)
    else: # randomly choose a node from the best node's neighbours
        candidate_idx = np.unique(np.random.RandomState(seed).choice(
            len(neighbors_of_best), batch_size)).tolist()
        candidates = neighbors_of_best[candidate_idx]
        # Now we add these new graph structure to ComboGraph for recording purposes
        center_combonode_to_stack = np.repeat(np.array(best_loc).reshape(1,-1),len(neighbors_of_best),axis=0)
        comboedges_array_to_add = np.stack((neighbors_of_best, center_combonode_to_stack),axis=1).astype(int)
        comboedges_to_add = list(tuple(map(tuple, i)) for i in comboedges_array_to_add) # Convert np.array to list of tuples
        ComboGraph.add_edges_from(comboedges_to_add) # record the explored structure
    return ComboGraph, X_train, Y_train, candidates
