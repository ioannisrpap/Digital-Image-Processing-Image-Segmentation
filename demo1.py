import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from spectral_clustering import spectral_clustering  
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


data = loadmat("dip_hw_3.mat")
affinity_matrix = data["d1a"]

colours_4 = ListedColormap(['red', 'darkorange', 'aqua', 'blueviolet'])  #note that these colours are just to differentiate the labels
#and have nothing to do with the actual colour there is 
cluster_colors = colours_4.colors  

patches = [mpatches.Patch(color=cluster_colors[i], label=f'Cluster {i}') for i in range(len(cluster_colors))]
for k in [2, 3, 4]: 
    labels = spectral_clustering(affinity_matrix, k)

    plt.figure()
    plt.title(f"Spectral clustering for k={k}")
    plt.imshow(labels.reshape(-1, 1), cmap=colours_4, aspect='auto')
    plt.legend(handles=patches, title="Clusters", bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()
