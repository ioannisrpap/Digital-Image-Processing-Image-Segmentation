import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

data = loadmat("dip_hw_3.mat")
datas = {
    "d2a": data["d2a"],
    "d2b": data["d2b"]
}
colours_4 = ListedColormap(['red', 'darkorange', 'aqua', 'blueviolet'])
cluster_colors = colours_4.colors  

patches = [mpatches.Patch(color=cluster_colors[i], label=f'Cluster {i}') for i in range(len(cluster_colors))]

for name, img in datas.items():
    M, N, _ = img.shape

    affinity = image_to_graph(img)

    for k in [2, 3, 4]:
        labels = spectral_clustering(affinity, k)
        
        clustered_img = labels.reshape(M, N)  

        plt.figure()
        plt.title(f"Spectral clustering for k={k}")
        plt.imshow(clustered_img, cmap=colours_4,aspect='auto' )
        plt.legend(handles=patches, title="Clusters", bbox_to_anchor=(1.00, 1), loc='upper left')
        

plt.show()
