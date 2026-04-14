import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh

def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:   #sthn mh anadromikh panta k=2

    D = np.diag(np.sum(affinity_mat, axis=1))
    L = D - affinity_mat

    eigvals, eigvecs = eigsh(L, k=k, M=D, which='SM')  #genikeumeno provlima L*x=lambda*D*x

    U = eigvecs

    kmeans = KMeans(n_clusters=k, random_state=1).fit(U)
    cluster_idx = kmeans.labels_

    return cluster_idx.astype(float)


