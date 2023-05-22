import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

np.random.seed(0)

################################ bk Means ################################
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


################################ Sammon Mapping ################################
def compute_gradient(Y, X):
    """
    Calculates the gradient for a given sammon mapping.

    Arguments:
        Y (numpy array): Config of points.
        data (numpy array): Original points.

    Return:
        numpy array: Gradient of sammon mapping.
    """
    s = Y.shape[0]
    distY = np.sqrt(np.sum((Y[:, np.newaxis] - Y) ** 2, axis=2))
    distD = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    grade = np.zeros(Y.shape)
    
    for i in range(s):
        tmpY = Y[i] - Y
        dataChange = X[i] - X
        tmpY[i] = 0
        dataChange[i] = 0
        # Be careful with division by zero.
        deriveE = (distY[i] - distD[i]) / ((distY[i] * distD[i]) + 1e-8)
        grade[i] = np.sum(tmpY * deriveE[:, np.newaxis], axis=0)

    return (grade * (2 / s))

def compute_stress(Y, X):
    """
    Computes stress for given sammon mapping.

    Arguments:
        Y (numpy array): Config of points.
        data (numpy array): Original points.

    Return:
        float: Current stress of sammon mapping.
    """
    distY = np.linalg.norm(Y[:, np.newaxis] - Y, axis=2)
    distD = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    stress = np.mean((distY - distD) ** 2)

    return stress

def sammon(X, iter=500, e=1e-7, a=.1):
    """
    Implements sammon mapping given a set of data and optionally 3 other parameters.

    Arguments:
        X (numpy array): Data to reduce, as an n x p matrix (n observations by p features).
        iter (int): Maximum number of iterations.
        e (float): Error threshold.
        a (float): Learning rate.
    
    Return:
        numpy array: A n x 2 vector with the final two-dimensional layout.
    """
    # 1. Start with a random two-dimensional layout Y of points (Y is a n × 2 matrix).
    Y = np.random.rand(X.shape[0], 2)

    # 2. Compute the stress E of Y.
    E = compute_stress(Y, X)

    # 3. If E < epsilon, or if the maximum number of iterations iterations has been reached, stop.
    for i in range(iter):
        if E < e:
            break
    # 4. For each yi of Y, find the next vector yi(t + 1) based on the current yi(t).
        gradient = compute_gradient(Y, X)
        Y -= a * gradient
        E = compute_stress(Y, X)
    
    return Y