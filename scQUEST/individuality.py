from dataclasses import dataclass, fields, field
from typing import Union, Callable, Iterable, get_args, Optional

import numpy as np
from anndata import AnnData

ArrayLike = np.ndarray

import pandas as pd

from scipy import sparse

from .utils import isCategorical

DistFunc = Callable[[np.ndarray, np.ndarray], float]
SparseMatrix = Union[sparse.csr_matrix, sparse.csr_matrix, sparse.csc_matrix]
Matrix = Union[np.ndarray, SparseMatrix]

from sklearn.neighbors import NearestNeighbors


@dataclass
class Individuality:
    """Computes the individuality of each observation in the data set according to [Wagner2019]_.

    Attributes:
        n_neighbors: number of neighbors in *k*\ NN graph
        radius: radius in radius graph
        graph_type: type of graph to build, either ``knn`` or ``radius``
        graph: if provided, uses this graph instead of construction an own one
        prior:
            either ``frequency``, ``uniform`` or custom prior probabilities for each class/group/label.
            If set to ``frequency`` the empirical class/group probabilities are used as prior.
            If set to ``uniform`` all classes/groups have the same prior probability.
        metric: distance metric to use when constructing the graph topology
        metric_params: additional kwargs passed to :attr:`~Individuality.metric`
        nn_params: additional kwargs passed to :class:`~.sklearn.NearestNeighbors`

    Notes:
        The posterior class probabilities for each observation are computed as follows [Bishop]_:

        .. math::

            p(c_i | x) &= \\frac{p(x | c_i) * p(c_i)}{p(x)} \\\\
            p(x | c_i) &= \\frac{K_i}{N_i * V} \\\\
            p(x) &= \\sum_{i \\in C}p(x | c_i)*p(c_i) \\\\
            p(c_i) &= \\frac{N_i}{N} \\;\\; \\texttt{if prior=frequency} \\\\
            p(c_i) &= \\frac{1}{|C|} \\;\\; \\texttt{if prior=uniform} \\\\

        Where

            - :math:`C` is the set of classes and :math:`c_i \\in C` a particular class
            - :math:`x` is the feature vector of an observation
            - :math:`K_i` the number of neighbor of class :math:`c_i`
            - :math:`V` the volume of the sphere that either

                - contains :attr:`~Individuality.n_neighbors` or
                - has radius :attr:`~Individuality.radius`.

            - :math:`N` the total number of observations and :math:`N_i` the number of observations belonging to class :math:`c_i`

     The resulting observation-level class probabilities estimates (:math:`N \\times |C|`) are aggregated (averaged) by each class
     to result in class-level individuality estimates (:math:`|C| \\times |C|`).

    Returns:
        Instance of :class:`Individuality`.

    """

    n_neighbors: Union[None, int] = 100
    radius: Union[None, float] = 1.0
    graph_type: str = "knn"
    graph: Union[None, Matrix] = None
    prior: Union[str, ArrayLike] = "frequency"
    metric: Union[DistFunc, str] = "minkowski"
    metric_params: dict = field(default_factory=dict)
    nn_params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nn_params = {"n_jobs": -1}
        self.NN = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            metric=self.metric,
            metric_params=self.metric_params,
            **self.nn_params,
        )

    def predict(
        self,
        ad: AnnData,
        labels: Union[Iterable, pd.Categorical],
        layer: Optional[str] = None,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """Performs prediction of the individuality of each observation and aggregates (average) results for each label.
        If you wish to access the posterior probabilities for each observation (cell) use :func:`~compute_individuality`.

        Args:
            X: matrix with observations as rows and columns as features.
            labels: indicator for the group/sample an observation belongs to.

        Returns:
            DataFrame with rows as observations and columns with the estimated probability to belong in the given group/sample

        """
        X = ad.X if layer is None else ad.layers[layer]
        uniq_labs = (
            labels.categories.values if isCategorical(labels) else np.unique(labels)
        )
        n_labs = len(uniq_labs)

        # convert to sequential, numerical labels starting at 0
        unique_numeric_sequential_labels = np.arange(n_labs)
        lab2num = {j: i for i, j in zip(unique_numeric_sequential_labels, uniq_labs)}
        num2lab = {i: j for i, j in zip(unique_numeric_sequential_labels, uniq_labs)}
        num_seq_labs = np.array([lab2num[i] for i in labels])

        self.graph = g = self._build_topology(X)

        posterior = self.compute_individuality(g, num_seq_labs, self.prior)

        # compute class-wise posterior mean
        # TODO: should we exclude entries with no neighbors, i.e. posterior.sum(1) == 0?
        # TODO: provide option for aggregation
        indices = [np.flatnonzero(num_seq_labs == i) for i in range(n_labs)]
        post_agg = np.vstack([posterior[idx].mean(axis=0) for idx in indices])
        # post_agg = np.vstack([np.median(posterior[idx], axis=0) for idx in indices])

        col_names = idx_names = [num2lab[i] for i in unique_numeric_sequential_labels]
        df = pd.DataFrame(post_agg, columns=col_names, index=idx_names)

        ad = ad if inplace else ad.copy()
        ad.obsm["individuality"] = posterior
        ad.uns["individuality_agg"] = df

    @staticmethod
    def compute_individuality(
        g: Matrix, num_seq_labs: ArrayLike, prior: Union[str, ArrayLike]
    ) -> np.ndarray:
        """
        Computes the observation-level individuality based on a given graph structure and labels.
        See :class:`Individuality` for an in-depth description.

        Args:
            g: graph encoding the observation-level interactions
            num_seq_labs: labels of the observations. Must be sequential and numeric, i.e. [0,1,2,...,N]
            prior:
                either ``frequency``, ``uniform`` or custom prior probabilities for each class/group/label.
                If set to ``frequency`` the empirical class/group probabilities are used as prior.
                If set to ``uniform`` all classes/groups have the same prior probability.

        Returns:
            :math:`N\\times |C|` :class:`~numpy.ndarray` with :math:`N` observations across :math:`|C|` classes.

        .. _link: https://medium.com/mlearning-ai/k-nearest-neighbor-knn-explained-with-examples-c32825fc9c43

        """
        n_labs = len(np.unique(num_seq_labs))

        # map edges to obs labels and count
        def map_and_count(row):
            return np.bincount(num_seq_labs[row == 1], minlength=n_labs)

        # NOTE: this could be optimised to work on the sparse matrix
        K_i = np.apply_along_axis(map_and_count, 1, g.A if sparse.issparse(g) else g)

        if prior == "frequency":
            posterior = K_i / g.sum(1).reshape(-1, 1)
        elif prior == "uniform":
            N_i = np.bincount(num_seq_labs)
            evidence = (K_i / N_i).sum(1).reshape(-1, 1)
            posterior = K_i / N_i / evidence
        elif isinstance(prior, ArrayLike):
            prior = np.asarray(prior)
            assert np.isclose(
                prior.sum(), 1
            ), f"prior probabilities do not sum to 1 but {prior.sum()}"

            N_i = np.bincount(num_seq_labs)
            evidence = (K_i / N_i * prior).sum(1).reshape(-1, 1)
            posterior = K_i * prior / N_i / evidence
        else:
            raise ValueError(
                f"Prior value {prior} is not valid. Either choose `frequency` or `uniform` or provide a vector with the class priors"
            )

        return posterior

    def _build_topology(self, x) -> Matrix:
        if self.graph is None:
            self.NN.fit(x)
            self.graph_builder = (
                self.NN.kneighbors_graph
                if self.graph_type == "knn"
                else self.NN.radius_neighbors_graph
            )
            return self.graph_builder()
        elif isinstance(self.graph, get_args(Matrix)):
            return self.graph
        else:
            raise TypeError(
                f"`graph` is of type {type(self.graph)} but should be (sparse) matrix."
            )

    def _check_args(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
