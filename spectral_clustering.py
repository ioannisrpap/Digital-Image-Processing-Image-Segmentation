import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh  
from image_to_graph import *

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    
    D = np.diag(np.sum(affinity_mat, axis=1)) 

    L = D - affinity_mat

    eigvals, eigvecs = eigsh(L, k=k, which='SM') #returns tuple, L from L*x=lambda*x, sm= small magnitude

    U = eigvecs

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    cluster_idx = kmeans.labels_

    return cluster_idx.astype(float)
