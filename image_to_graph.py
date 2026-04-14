import numpy as np
def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    
    M, N, C = img_array.shape
    nodes = M * N     #every pixel is a node 

    reshaped_img = img_array.reshape((nodes, C))  #make it flat with vectors in R^c

    sum_of_sq = np.sum(reshaped_img**2, axis=1, keepdims=True)    #calculates ||xi||^2 for every pixel
    
    dist_sq = sum_of_sq + sum_of_sq.T - 2 * np.dot(reshaped_img, reshaped_img.T) # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 * xi*xj
    
    dist_sq = np.maximum(dist_sq, 0)  #just to fix a possible small negative number by 'akriveia'

    dist = np.sqrt(dist_sq)  #remove the ^2

     
    A = 1.0 / np.exp(dist) #A(i,j) = 1 / e^d(i,j)

    return A   #tetragonos kai symmetrikos

