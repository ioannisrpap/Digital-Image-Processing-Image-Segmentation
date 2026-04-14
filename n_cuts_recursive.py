import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from calculate_n_cut_value import *
from n_cuts import *
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph


    
    #made a better version gia na exoun ola ta fylla ena label kai oxi endiamesa labels 
def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    
    nodes = affinity_mat.shape[0]
    cluster_idx = np.full(nodes, -1, dtype=float)  
    current_label = 0  

    def recursive_split(nodes):
        nonlocal current_label

        if len(nodes) < T1:
            cluster_idx[nodes] = current_label
            current_label += 1
            return

        sub_affinity = affinity_mat[np.ix_(nodes, nodes)]
        labels = n_cuts(sub_affinity, 2)
        ncut_val = calculate_n_cut_value(sub_affinity, labels)

        clusterA_nodes = nodes[labels == 0]
        clusterB_nodes = nodes[labels == 1]

        if len(clusterA_nodes) < T1 or len(clusterB_nodes) < T1 or ncut_val > T2:
            cluster_idx[nodes] = current_label
            current_label += 1
            return

        recursive_split(clusterA_nodes)
        recursive_split(clusterB_nodes)

    all_nodes = np.arange(nodes)
    recursive_split(all_nodes)

    unique_labels = np.unique(cluster_idx)
    for new_label, old_label in enumerate(unique_labels):
        cluster_idx[cluster_idx == old_label] = new_label

    return cluster_idx
