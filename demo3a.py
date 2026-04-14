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
from n_cuts_recursive import *




data = loadmat("dip_hw_3.mat")
images = {
    "d2a": data["d2a"].squeeze(),
    "d2b": data["d2b"].squeeze()
}

for name, img in images.items():
    M, N, _ = img.shape
    affinity = image_to_graph(img)

    for k in [2, 3, 4]:
        
        labels_nc = n_cuts(affinity, k)
        clustered_nc = labels_nc.reshape(M, N)

        labels_sc = spectral_clustering(affinity, k)
        clustered_sc = labels_sc.reshape(M, N)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"n-cuts k={k} on {name}")
        plt.imshow(clustered_nc, cmap="tab20")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Spectral Clustering k={k} on {name}")
        plt.imshow(clustered_sc, cmap="tab20")
        plt.axis('off')

        plt.show()



