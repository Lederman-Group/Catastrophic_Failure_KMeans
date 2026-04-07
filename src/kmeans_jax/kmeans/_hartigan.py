from typing import Tuple

import numpy as np
from jaxtyping import Array, Float, Int
from numba import jit


@jit
def _compute_loss_np(data, centroids, labels):
    return np.sum(np.abs(data - centroids[labels]) ** 2)


@jit
def _assign_labels_lloyd_np(centroids, data):
    distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=1)


@jit
def _compute_centroids_np(data, labels, centroids):
    for i in range(centroids.shape[0]):
        idx_mask = labels == i
        n_elements = np.sum(idx_mask)
        if n_elements > 0:
            centroids[i] = np.sum(data[idx_mask], axis=0) / n_elements
        else:
            centroids[i] = np.zeros_like(centroids[i])
    return centroids


@jit
def _assign_label_hartigan_np(centroids, cluster_populations, data_point, label_point):
    distances = np.sum((data_point[None, ...] - centroids) ** 2, axis=-1)

    for i in range(centroids.shape[0]):
        if label_point == i:
            if cluster_populations[i] <= 1:
                distances[i] = -1.0 # always assign
            else:
                scale_factor = cluster_populations[i] / (cluster_populations[i] - 1)
                distances[i] *= scale_factor

        else:
            scale_factor = cluster_populations[i] / (cluster_populations[i] + 1)
            distances[i] *= scale_factor
    return np.argmin(distances)


@jit
def _run_hartigan_numpy(data, init_centroids, max_iters):

    # Initial quantities
    labels = _assign_labels_lloyd_np(init_centroids, data)
    centroids = _compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    # Variables to update
    old_labels = labels.copy()
    n_iters = 0
    for n_iters in range(max_iters):
        for j in range(data.shape[0]):
            new_label = _assign_label_hartigan_np(
                centroids, cluster_populations, data[j], labels[j]
            )
            if new_label != labels[j]:
                # centroids = compute_centroids(data, labels, centroids)
                n_clust = cluster_populations[labels[j]]
                centroids[labels[j]] = (centroids[labels[j]] * n_clust - data[j]) / (
                    n_clust - 1.0
                )

                n_clust = cluster_populations[new_label]
                centroids[new_label] = (centroids[new_label] * n_clust + data[j]) / (
                    n_clust + 1.0
                )

                cluster_populations[labels[j]] -= 1
                cluster_populations[new_label] += 1
                labels[j] = new_label

        if np.array_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()

    loss = _compute_loss_np(data, centroids, labels)
    return centroids, labels, loss, n_iters


def run_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    """
    Run k-means clustering using online Hartigan's algorithm. Unlike other algorithms
    in the library, this is implemented to run on Numba, and thus has no GPU support.

    **Arguments**:
        data: A numpy-like array of shape (n, d) containing the data points.
        init_centroids: A numpy-like array of shape (K, d)
                        containing the initial centroids.
        max_iters: maximum number of iterations.

    **Returns**:
        A tuple containing:
            - centroids: A numpy-like array of shape (K, d) containing the final centroids
            - labels: A numpy-like array of shape (n,) containing the final cluster labels
            - loss: A float representing the final k-means loss.
            - n_iters: An integer representing the number of iterations performed.
    """
    return _run_hartigan_numpy(
        np.asanyarray(data), np.asanyarray(init_centroids), max_iters
    )
