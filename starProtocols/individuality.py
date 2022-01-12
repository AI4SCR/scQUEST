from dataclasses import dataclass, fields
from typing import Union, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .utils import Predictor
from .utils import isCategorical

DistFunc = Callable[[np.ndarray, np.ndarray], float]
SparseMatrix = Union[sparse.csr_matrix, sparse.csr_matrix, sparse.csc_matrix]

from sklearn.neighbors import NearestNeighbors


@dataclass
class Individuality:
    """Computes the individuality of each cell in the data set.
    https://medium.com/mlearning-ai/k-nearest-neighbor-knn-explained-with-examples-c32825fc9c43"""

    n_neighbors: Union[None, int] = 100
    radius: Union[None, float] = 1.0
    graph: Union[str, SparseMatrix] = 'knn'
    prior: str = 'frequency'
    metric: Union[DistFunc, str] = 'minkowski'
    metric_params: dict = None,
    nn_params: dict = None

    def __post_init__(self):
        self._check_args()

        self.NN = NearestNeighbors(n_neighbors=self.n_neighbors, radius=self.radius,
                                   metric=self.metric, metric_params=self.metric_params,
                                   **self.nn_params)

    def _predict(self, x, y: Union[Iterable, pd.Categorical] = None):
        uniqLabs = y.categories.values if isCategorical(y) else np.unique(y)
        nlabs = len(uniqLabs)

        # convert to sequential, numerical labels starting at 0
        numLabs = np.arange(nlabs)
        lab2num = {j: i for i, j in zip(numLabs, uniqLabs)}
        num2lab = {i: j for i, j in zip(numLabs, uniqLabs)}
        labs = np.array([lab2num[i] for i in y])

        g = self._build_topology(x)
        g[np.diag_indices_from(g)] = 0  # remove self-edges

        # map edges to obs labels and count
        def map_and_count(x):
            return np.bincount(labs[x == 1], minlength=nlabs)

        K_k = np.apply_along_axis(map_and_count, 1, g)

        if self.prior == 'frequency':
            posterior = K_k / g.sum(1).reshape(-1,1)
        elif self.prior == 'uniform':
            N_k = np.bincount(labs)
            evidence = (K_k / N_k).sum(1).reshape(-1,1)
            posterior = K_k / N_k / evidence
        else:
            N_k = np.bincount(labs)
            prior = np.ones_like(numLabs) / nlabs
            evidence = (K_k / N_k * prior).sum(1).reshape(-1,1)
            posterior = K_k * prior / N_k / evidence

        # NOTE: if two classes have the same posterior, the current implementation choses the smaller class label
        #  as the argmax value
        argmax_class = np.argmax(posterior, axis=1)

        # compute class-wise posterior mean
        indices = [np.flatnonzero(labs == i) for i in range(nlabs)]
        post_mean = np.vstack([posterior[idx].mean(axis=0) for idx in indices])

        df = pd.DataFrame(post_mean, columns=[num2lab[i] for i in numLabs])

        return df

    def predict(self, x, y: Union[Iterable, pd.Categorical] = None):
        uniqLabs = y.categories.values if isCategorical(y) else np.unique(y)
        nlabs = len(uniqLabs)

        # convert to sequential, numerical labels starting at 0
        numLabs = np.arange(nlabs)
        lab2num = {j: i for i, j in zip(numLabs, uniqLabs)}
        num2lab = {i: j for i, j in zip(numLabs, uniqLabs)}
        labs = np.array([lab2num[i] for i in y])

        g = self._build_topology(x)
        g[np.diag_indices_from(g)] = 0  # remove self-edges

        # map edges to obs labels and count
        def map_and_count(x):
            return np.bincount(labs[x == 1], minlength=nlabs)

        counts = np.apply_along_axis(map_and_count, 1, g)

        # compute probabilities
        rowSum = counts.sum(1).reshape(-1, 1)
        likelihood = np.divide(counts, rowSum, where=rowSum != 0, out=np.zeros_like(counts, dtype=float))

        if self.prior == 'frequency':
            prior = np.bincount(numLabs) / nlabs
        else:
            # uniform
            prior = np.ones_like(likelihood) / nlabs

        evidence = (prior * likelihood).sum(1).reshape(-1, 1)
        posterior = likelihood * prior / evidence

        # NOTE: if two classes have the same posterior, the current implementation choses the smaller class label as the argmax value
        argmax_class = np.argmax(posterior, axis=1)

        # compute class-wise posterior mean
        indices = [np.flatnonzero(numLabs == i) for i in range(nlabs)]
        post_mean = np.vstack([posterior[idx].mean(axis=0) for idx in indices])

        pd.DataFrame(post_mean, columns=[num2lab[i] for i in numLabs])

        return post_mean

    def _build_topology(self, x):
        if isinstance(self.graph, str):
            self.NN.fit(x)
            self.graph_builder = self.NN.kneighbors_graph() if self.graph == 'knn' else self.NN.radius_neighbors_graph()
            return self.graph_builder(x)
        else:
            return self.graph

    def _check_args(self):
        for field in fields(self):
            value = getattr(self, field.name)
