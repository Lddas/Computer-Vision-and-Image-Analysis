import numpy as np


def kmeans_fit(data, k, n_iter=500, tol=1.e-4):
    """
    Fit kmeans
    
    Args:
        data ... Array of shape n_samples x n_features
        k    ... Number of clusters
        
    Returns:
        centers   ... Cluster centers. Array of shape k x n_features
    """
    N, _ = data.shape
    
    # Create a random number generator
    # Use this to avoid fluctuation in k-means performance due to initialisation
    rng = np.random.default_rng(6174)
    
    # Initialise clusters
    centroids = data[rng.choice(N, k, replace=False)]
    
    # Iterate the k-means update steps
    #
    # TO IMPLEMENT

    for _ in range(n_iter):
      idx = np.argmin(compute_distance(data,centroids), axis=1)
      new_centroids = np.array([data[idx == i].mean(axis=0) for i in range(k)])

      if np.all(np.abs(new_centroids - centroids) < tol):
          break

      centroids = new_centroids
            
    # Return cluster centers
    return centroids


def compute_distance(data, clusters):
    """
    Compute all distances of every sample in data, to every center in clusters.
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
        
    Returns:
        distances ... n_samples x n_clusters
    """

    '''rows = len(data)
    cols = len(clusters)
    distances = np.empty((rows, cols))
    for i, x in enumerate(data):
      for j, centroid in enumerate(clusters):
        d = np.sqrt(np.sum((x - centroid) ** 2))
        distances[i][j] = d'''

    distances = np.linalg.norm(data[:, np.newaxis] - clusters, axis=2)
    return distances


def kmeans_predict_idx(data, clusters):
    """
    Predict index of closest cluster for every sample
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
    """
    # TO IMPLEMENT
    idx = np.argmin(compute_distance(data,clusters), axis=1)
    idx = idx.reshape(data.shape[0])
    return idx