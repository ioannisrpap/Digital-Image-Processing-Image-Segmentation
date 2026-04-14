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

    labels = n_cuts(affinity, k=2)
    clustered = labels.reshape(M, N)

    ncut_val = calculate_n_cut_value(affinity, labels)

    plt.figure(figsize=(6, 5))
    plt.title(f"n-cuts step (k=2) on {name}\nNcut value: {ncut_val:.4f}")
    plt.imshow(clustered, cmap="tab20")
    plt.axis('off')
    plt.show()





