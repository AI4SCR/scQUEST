from dataclasses import dataclass, fields
from typing import Union, Callable

import numpy as np
from scipy import sparse

from .utils import Predictor

DistFunc = Callable[[np.ndarray, np.ndarray], float]
SparseMatrix = Union[sparse.csr_matrix, sparse.csr_matrix, sparse.csc_matrix]

from sklearn.neighbors import NearestNeighbors


@dataclass
class Individuality(Predictor):
    """Computes the individuality of each cell in the data set."""

    n_neighbors: Union[None, int] = 100
    radius: Union[None, float] = 1.0
    graph: Union[str, SparseMatrix] = 'knn'
    prior: str = 'uniform'
    metric: Union[DistFunc, str] = 'minkowski'
    metric_params: dict = None,
    nn_params: dict = None

    def __post_init__(self):
        self._check_args()

        self.NN = NearestNeighbors(n_neighbors=self.n_neighbors, radius=self.radius,
                                   metric=self.metric, metric_params=self.metric_params,
                                   **self.nn_params)
        self.graph_builder = self.NN.kneighbors_graph() if self.graph_type == 'knn' else self.NN.radius_neighbors_graph()

    def fit(self, x, y=None):
        # TODO: do not fit when a graph is provided
        self.NN.fit(x)
        self.g = self.graph_builder(x)

    def predict(self, x, y=None):
        pass

    def _compute_prior(self):
        pass

    def _build_topology(self):
        pass

    def _check_args(self):
        for field in fields(self):
            value = getattr(self, field.name)
