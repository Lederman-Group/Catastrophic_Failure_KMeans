# The Catastrophic Failure of the k-Means Algorithm in High Dimensions, and How Hartigan's Algorithm Avoids It

## Summary

This repository provides the library code and scripts necessary to reproduce the results of our paper, currently available as a preprint at: TBA. The zip file `scripts_and_results.zip` provides pre-computed results, the scripts necessary to obtain them, and Jupyter Notebooks for their visualization.

The library code implements Lloyd's and Hartigan's k-Means in JAX and Numba, respectively.

## Installation

The package can be installed with pip. If using a GPU for computations, install jax first

```bash
pip install jax[cuda12]
```

Then install the package

```bash
python -m pip install .
```

## Tutorial

Here we provide a short tutorial for running k-Means with out library using a similar API to SciKit Learn:

```python
import kmeans_jax
import jax.random as jr

data = ... # data with shape (n_data, dim_data)

kmeans = kmeans_jax.KMeans(
    n_clusters=...,
    n_init=...,
    max_iter=...,
    init=..., # one of 'random', 'random partition' or 'kmeans++'
    algorithm=..., # one of 'Hartigan' or 'Lloyd'
)

kmeans_results = kmeans.fit(
    key=jr.key(seed),
    data,
    output="best", # outputs the best output in terms of the k-means loss
)
# If you want the outputs for all initializations, then set output = 'all'

print(kmeans_results) # if output='all' the outputs will have a batch dimension

>> dict(centroids=..., labels=..., loss=...)
```


## Acknowledgements

(suppressed for Anonymous submission)
