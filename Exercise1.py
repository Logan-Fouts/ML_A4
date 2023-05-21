import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans


def sse(cluster):
    """
    Calculates the SSE for given cluster.

    Params:
        cluster (np array): The cluster in which to find the SSE.
    
    Return:
        float: SSE value for the cluster.
    """
    clustMean = np.mean(cluster, axis=0)
    squareErr = (cluster - clustMean) ** 2
    sse = np.sum(squareErr)

    return sse


def bkmeans(X, k, iter):
    """
    Executes Bisecting k-means clustering with provided arguments.

    Parameters:
        X (numpy array): Data to perform clustering on.
        k (int): Number of differnt clusters to seperate out.
        iter (int): Number of times to iterate inside of K means.
    Return:
        numpy array: n x 1 vector with cluster indices for each of the observations.
    """
    # Single cluster containing all observations.
    clusters = [X]

    # Bisecting k-Means.
    while len(clusters) < k:
        # Find cluster with the largest SSE.
        largestSSE_cluster = clusters[0]
        largestSSE = sse(clusters[0])

        for cluster in clusters[1:]:
            tmpSSE = sse(cluster)
            if tmpSSE > largestSSE:
                largestSSE = tmpSSE
                largestSSE_cluster = cluster
        
        # K-Means on the cluster from above.
        kmeans = KMeans(n_clusters=2, n_init=iter)
        kmeans.fit(largestSSE_cluster)
        labels = kmeans.labels_
        
        # Split into two clusters.
        subClust1 = largestSSE_cluster[labels == 0]
        subClust2 = largestSSE_cluster[labels == 1]
        
        # Remove largest cluster add two subclusters.
        tmp_clusters = []
        for cluster in clusters:
            if not np.array_equal(cluster, largestSSE_cluster):
                tmp_clusters.append(cluster)
        clusters = tmp_clusters
        clusters.append(subClust1)
        clusters.append(subClust2)

    # Assign cluster indices for data points.
    cluster_indices = np.zeros(X.shape[0])
    for i, cluster in enumerate(clusters):
        indices = np.isin(X, cluster).all(axis=1)
        cluster_indices[indices] = i

    return cluster_indices.reshape([-1, 1])


################################ Testing ################################
iris = datasets.load_iris()
X = iris.data

k = 3
iterations = 10
cluster_indices = bkmeans(X, k, iterations)

plt.scatter(X[:, 0], X[:, 1], c=cluster_indices)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('bkmeans Clustering')
plt.show()
