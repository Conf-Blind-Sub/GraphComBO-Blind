import numpy as np
import networkx as nx
import random
import torch
import itertools
import multiprocessing
from math import comb
from functools import partial
from time import time

# Find the neighbors of a given ComboNode derived from the underlying Graph
def NeighborFinder(Graph, ComboNode): # Output edges connected to this ComboNode
    neighbors = np.empty(shape=(0,len(ComboNode)),dtype=int) # Initialise an empty neighbor np.array
    for idx, i in enumerate(ComboNode):
        ComboNode_list = np.delete(ComboNode,idx) # Get the elements except i
        # Find node i's neighbors in the original graph excluding nodes from ComboNode_list
        #i_original_neighbors = np.array([j for j in list(Graph.neighbors(i)) if j not in ComboNode_list])
        i_neighbors_set = set(list(Graph.neighbors(i))) - set(ComboNode_list.tolist())
        if len(i_neighbors_set)>0: # if node i has neighbors outside the current combination
            # Construct the combinatorial nodes for node i
            i_combo_neighbors = np.hstack((np.repeat(ComboNode_list.reshape(1,-1), len(i_neighbors_set), axis=0), # ComboNode without i
                                             np.array(list(i_neighbors_set)).reshape(-1,1) )) # i's neighbors in the original graph
            i_combo_neighbors.sort(axis=1) # sort the array to create a unique identifier for each combo node
            neighbors = np.vstack((neighbors, i_combo_neighbors))
        '''
        i_combo_neighbors = np.array(list(map(lambda x : np.append(ComboNode_list, x), i_original_neighbors)))
        if len(i_combo_neighbors)!=0: # stack the neighbors if it is not none
            i_combo_neighbors.sort(axis=1) # sort the array to create a unique identifier for each combo node
            neighbors = np.vstack((neighbors, i_combo_neighbors))
        '''
    # Construct the edges from this ComboNode to its neighbors
    stack = np.repeat(np.array(ComboNode).reshape(1,-1),len(neighbors),axis=0)
    edge_array = np.stack((neighbors, stack),axis=1)
    edges = list(tuple(map(tuple, i)) for i in edge_array) # Convert np.array to list of tuples
    return edges
            
# check if the edge exists in the current combonode-pair
def ComboEdge_Indicator(ComboNode_Pairs, A, idx): 
    difference = list(set(ComboNode_Pairs[idx][0]).symmetric_difference(set(ComboNode_Pairs[idx][1])))
    return True if len(difference)==2 and A[difference[0],difference[1]] else False

# Find the ComboSubgraph of size Q centred at a given ComboNode by gradually including multi-hops of neighbors
# Initialise the Combo_Subgraph as nx.Graph() and ComboNode_list as [ComboNode] at beginning 
def ComboSubgraph_Constructor(Graph, ComboNode_list, ComboSubgraph, Q=200, l=1, l_max=100, large_Q_computation=False, start_time=None): 
    start_time = time() if l==1 else start_time # record the time for computing current ComboSubgraph
    previous_ComboSubgraph_set = set(ComboSubgraph.nodes()) if l>1 else set([tuple(ComboNode_list[0])])
    '''
    def map_func(ComboNode):
        return NeighborFinder(Graph, ComboNode)
    ComboEdges_list = np.vstack(list(map(map_func,ComboNode_list)))
    edges = list(tuple(map(tuple, i)) for i in edge_array) # Convert np.array to list of tuples
    ComboSubgraph.add_edges_from(ComboEdges_list)
    '''
    for ComboNode in ComboNode_list: # add new neighbors of the previous neighbors to the previous ComboSubgraph
        ComboSubgraph.add_edges_from(NeighborFinder(Graph, ComboNode))
        if ComboSubgraph.number_of_nodes()>=Q and large_Q_computation:
            break
    new_neighbors = list(set(ComboSubgraph.nodes()) - previous_ComboSubgraph_set)
    random.shuffle(new_neighbors)
    print(f"number of new neighbors at hop {l}: {len(new_neighbors)}")
    if ComboSubgraph.number_of_nodes()<=Q and len(new_neighbors)>0 and l<l_max: # do a recursive operation if ComboSubgraph's size < Q
        # Here we set all the neighbors at the current hop as the ComboNode_list for next step
        return ComboSubgraph_Constructor(Graph, new_neighbors, ComboSubgraph, Q, l+1, large_Q_computation=large_Q_computation, start_time=start_time)
    else: # randomly remove a subset of neighbors at the current hop to reach Q 
        if len(ComboSubgraph)>Q:
            nodes_to_remove = random.sample(new_neighbors, 
                                            k=ComboSubgraph.number_of_nodes()-Q)
            ComboSubgraph.remove_nodes_from(nodes_to_remove)
        if l==1 or large_Q_computation: # Add edges among the ComboNodes
            # From definition in paper: compare if only 1 out of k nodes in each combonode 
            # is different from another combonode across all combonodes in the combosubgraph
            d_type = np.ushort if len(Graph)<=65535 else np.unit32 # use unsigned int type to save memory
            A = nx.adjacency_matrix(Graph).todense()
            X = np.array(list(ComboSubgraph.nodes)).astype(d_type)
            X1, X2 = X[:,None,:], X[None,:,:] # X1.shape=[Q,1,k], X2.shape=[1,Q,k]
            X_concat = np.concatenate([np.repeat(X1, X.shape[0], axis=1), # [Q,Q,k]
                                       np.repeat(X2, X.shape[0], axis=0)], # [Q,Q,k]
                                      axis=2) # Concat all the combonode-pairs for later computation
            # Now we check if only 1 element is different in each combonode-pair
            # The resulting matrix is a symmetric [Q, Q] adj matrix of True/False
            X_concat.sort(axis=2, kind='mergesort') # we sort first and then take the difference: i+1 - i
            raw_adj = np.sum(np.diff(X_concat,axis=2) != 0, axis=2) == X.shape[-1]
            # get the [row,col] idx of True in the upper triangular part.
            raw_edges_idx = np.argwhere(np.triu(raw_adj)) 
            # Now we check if these comboedges actually exist by looking at the original graph
            raw_edges = X[raw_edges_idx]
            partial_function = partial(ComboEdge_Indicator, raw_edges, A)
            edges_bool = list(map(partial_function, range(len(raw_edges))))
            edge_tuples = list(tuple(map(tuple, i)) for i in raw_edges[edges_bool]) # convert them to tuples
            ComboSubgraph.add_edges_from(edge_tuples)
        print(f'Current size of the ComboSubgraph: {len(ComboSubgraph)}. Time for computation: {(time()-start_time):.1f}s')
        return ComboSubgraph, l

# Construct the ENTIRE ComboGraph for k-node combinations given an underlying Graph.
# Do NOT try this with large combo-graph, memory explodes!
def ComboGraph_Constructor(Graph,k,node=None): 
    ComboGraph = nx.Graph() # Initialise the ComboGraph
    total_comb = comb(Graph.number_of_nodes(), k)
    print("Number of ComboNodes:", total_comb)
    if node == None:
        nodes_selfcombo = list(itertools.combinations(list(Graph.nodes()),k)) # All the possible ComboNodes
        # Use multiprocessing to parallel compute each ComboNode's edges
        partial_function = partial(NeighborFinder, Graph)
        with multiprocessing.Pool(8) as pool:
            ComboEdges = pool.map(partial_function, nodes_selfcombo)
    for x in ComboEdges:
        ComboGraph.add_edges_from(x)
    print("Number of ComboEdges:", ComboGraph.number_of_edges())
    return ComboGraph, total_comb

def NeighborFinder_New(Graph, ComboNode): # Output edges connected to this ComboNode
    neighbors = np.empty(shape=(0,len(ComboNode)),dtype=int) # Initialise an empty neighbor np.array
    def map_func(idx):
        ComboNode_list = np.delete(ComboNode,idx) # Get the elements except i
        # Find node i's neighbors in the original graph excluding nodes from ComboNode_list
        i_neighbors_set = set(list(Graph.neighbors(ComboNode[idx]))) - set(ComboNode_list.tolist())
        if len(i_neighbors_set)>0: # if node i has neighbors outside the current combination
            # Construct the combinatorial nodes for node i
            i_combo_neighbors = np.hstack((np.repeat(ComboNode_list.reshape(1,-1), len(i_neighbors_set), axis=0), # ComboNode without i
                                             np.array(list(i_neighbors_set)).reshape(-1,1) )) # i's neighbors in the original graph
            i_combo_neighbors.sort(axis=1) # sort the array to create a unique identifier for each combo node
            return i_combo_neighbors
        else:
            pass
    result = list(map(map_func, range(len(ComboNode))))
    neighbors = np.vstack((neighbors, i_combo_neighbors))
    __import__("pdb").set_trace()
    # Construct the edges from this ComboNode to its neighbors
    stack = np.repeat(np.array(ComboNode).reshape(1,-1),len(neighbors),axis=0)
    edge_array = np.stack((neighbors, stack),axis=1)
    edges = list(tuple(map(tuple, i)) for i in edge_array) # Convert np.array to list of tuples
    return edges

def ComboSubgraph_Constructor_Old(Graph, ComboNode_list, ComboSubgraph, Q=200, l=1, L_max=None, large_Q_computation=False): 
    previous_ComboSubgraph_set = set(ComboSubgraph.nodes()) if l>1 else set([tuple(ComboNode_list[0])])
    '''
    def map_func(ComboNode):
        return NeighborFinder(Graph, ComboNode)
    ComboEdges_list = np.vstack(list(map(map_func,ComboNode_list)))
    edges = list(tuple(map(tuple, i)) for i in edge_array) # Convert np.array to list of tuples
    ComboSubgraph.add_edges_from(ComboEdges_list)
    '''
    for ComboNode in ComboNode_list: # add new neighbors of the previous neighbors to the previous ComboSubgraph
        ComboSubgraph.add_edges_from(NeighborFinder(Graph, ComboNode))
        if ComboSubgraph.number_of_nodes()>=Q and large_Q_computation:
            break
    new_neighbors = list(set(ComboSubgraph.nodes()) - previous_ComboSubgraph_set)
    random.shuffle(new_neighbors)
    print(f"number of new neighbors at hop {l}: {len(new_neighbors)}")
    if ComboSubgraph.number_of_nodes()<=Q and len(new_neighbors)>0: # do a recursive operation if ComboSubgraph's size < Q
        # Here we set all the neighbors at the current hop as the ComboNode_list for next step
        return ComboSubgraph_Constructor(Graph, new_neighbors, ComboSubgraph, Q, l+1, large_Q_computation=large_Q_computation)
    else: # randomly remove a subset of neighbors at the current hop to reach Q 
        nodes_to_remove = random.sample(new_neighbors, 
                                        k=ComboSubgraph.number_of_nodes()-Q)
        ComboSubgraph.remove_nodes_from(nodes_to_remove)
        t1 = time()
        if large_Q_computation: # Add edges among the ComboNodes
            A = nx.adjacency_matrix(Graph).todense()
            # get all the possible combonode-pairs in the subgraph
            ComboNode_Pairs = list(itertools.combinations(list(ComboSubgraph.nodes()), 2))
            partial_function = partial(ComboEdge_Indicator, ComboNode_Pairs, A)
            #with multiprocessing.Pool() as pool:
                #edges_bool = pool.map(partial_function, range(len(ComboNode_Pairs)))
            edges_bool = list(map(partial_function,range(len(ComboNode_Pairs))))
            edge_array = np.array(ComboNode_Pairs)[edges_bool] # get the comboedges 
            edge_tuples = list(tuple(map(tuple, i)) for i in edge_array) # convert them to tuples
            ComboSubgraph.add_edges_from(edge_tuples)
            if False:
                # From definition in paper: compare if only 1 out of k nodes in each combonode 
                # is different from another combonode across all combonodes in the combosubgraph
                d_type = np.ushort if len(Graph)<=65535 else np.unit32 # use unsigned int type to save memory
                X = np.array(list(ComboSubgraph.nodes)).astype(d_type)
                X1, X2 = X[:,None,:], X[None,:,:] # X1.shape=[Q,1,k], X2.shape=[1,Q,k]
                X_concat = np.concatenate([np.repeat(X1, X.shape[0], axis=1), # [Q,Q,k]
                                           np.repeat(X2, X.shape[0], axis=0)], # [Q,Q,k]
                                          axis=2) # Concat all the combonode-pairs for later computation
                # Now we check if only 1 element is different in each combonode-pair
                # The resulting matrix is a symmetric [Q, Q] adj matrix of True/False
                X_concat.sort(axis=2) # we sort first and then take the difference: i+1 - i
                raw_adj = np.sum(np.diff(X_concat,axis=2) != 0, axis=2) == X.shape[-1]
                # get the [row,col] idx of True in the upper triangular part.
                raw_edges_idx = np.argwhere(np.triu(raw_adj)) 
                # Now we check if these comboedges actually exist by looking at the original graph
                raw_edges = X[raw_edges_idx]
                partial_function = partial(ComboEdge_Indicator, raw_edges, A)
                edges_bool = list(map(partial_function, range(len(raw_edges))))
                edge_tuples = list(tuple(map(tuple, i)) for i in raw_edges[edges_bool]) # convert them to tuples
                ComboSubgraph.add_edges_from(edge_tuples)
        return ComboSubgraph, l
