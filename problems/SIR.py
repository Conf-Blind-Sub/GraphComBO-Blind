import numpy as np
import torch
import networkx as nx
import multiprocessing
import time
import warnings
import ndlib.models.ModelConfig as mc
from ndlib.models.epidemics.SIRModel import SIRModel

def get_initial_status(Graph, ComboNode, mapping): # set the initial state to the chosen ComboNode
    node_list = [i for i in Graph.nodes()]
    for node in node_list:
        if node in ComboNode:
            mapping[node] = 2
    return mapping

def One_SIR_Run(graph, cfg, ComboNode, T, idx):
    model = SIRModel(graph, idx)
    model.set_initial_status(cfg)
    model.status = get_initial_status(graph, ComboNode, model.status)
    iterations = model.iteration_bunch(T)
    return iterations

def SIR_MC(T,N,n_samples,threshold,parallel_function,parallel=True):
    # Simulation execution
    if parallel:
        with multiprocessing.Pool() as pool:
            results = pool.map(parallel_function, list(range(n_samples)))
    else:
        results = list(map(parallel_function,list(range(n_samples))))
    infection_time_list = []
    for j in range(n_samples):
        S = [results[j][i]['node_count'][0] for i in range(T)]
        infection_time = sum(np.array(S)>(N*(1-threshold)))
        infection_time_list.append(infection_time)
    infection_time_list = np.array(infection_time_list)
    infection_time_valid = infection_time_list[~(infection_time_list == T)] # filter out invalid simulations
    function_value = np.mean(infection_time_valid).round(1)
    return function_value # optional: make it negative if we want to minimize it in the experiment.
