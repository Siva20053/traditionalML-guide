import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from Kmeans import KMeans

data = load_iris()
X = data.data[:, 2:4]  

kmeans = KMeans(K=3, max_iters=100)
kmeans.fit(X)

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['red','green','blue'] 

for i, cluster_idxs in enumerate(kmeans.clusters):
    cluster_points = X[cluster_idxs]
    
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               color=colors[i], label=f'Cluster {i+1}', alpha=0.6, s=50)

centroids = np.array(kmeans.centroids)
ax.scatter(centroids[:, 0], centroids[:, 1], 
           marker="x", color="black", s=200, linewidth=3, label='Centroids')

ax.set_title("K-Means Clustering on Iris Data (Petal Dimensions)")
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.legend()
plt.grid(True, alpha=0.3)
plt.show()