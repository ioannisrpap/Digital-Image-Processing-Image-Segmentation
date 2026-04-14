import numpy as np

def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
 
    cluster_A = [i for i in range(len(cluster_idx)) if cluster_idx[i] == 0]
    cluster_B = [i for i in range(len(cluster_idx)) if cluster_idx[i] == 1]

    def assoc(X, Y):    
        return np.sum(affinity_mat[np.ix_(X, Y)]) #edges starting from 1 cluster ending to the 

    assoc_A_V = assoc(cluster_A, list(range(len(cluster_idx))))
    assoc_B_V = assoc(cluster_B, list(range(len(cluster_idx))))

    assoc_A_A = assoc(cluster_A, cluster_A)
    assoc_B_B = assoc(cluster_B, cluster_B)

    n_cut_value = 2 - (assoc_A_A / assoc_A_V + assoc_B_B / assoc_B_V)

    return n_cut_value



