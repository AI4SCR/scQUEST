"""Unit testing for module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import string
from itertools import product

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scQUEST import Individuality


def get_n_groups_label(n_obs_per_group: list = [5, 5]):
    labs = np.arange(len(n_obs_per_group), dtype=int)
    labs = np.repeat(labs, n_obs_per_group)
    return labs


def get_n_groups_feat(n_obs_per_group: list = [5, 5]):
    n_groups = len(n_obs_per_group)
    n_feat = max(np.ceil(np.log2(n_groups)).astype(int), 1)
    feat_generator = product(*[[0, 1]] * n_feat)
    feat = np.array([next(feat_generator) for i in range(n_groups)])
    feat = np.repeat(feat, n_obs_per_group, axis=0)

    return feat


def get_n_groups_data(n_obs_per_group: list = [5, 5]):
    return get_n_groups_feat(n_obs_per_group), get_n_groups_label(n_obs_per_group)


class TestIndividuality:
    @pytest.mark.parametrize("n_obs_per_group", [[2], [2, 2], [5, 5], [3, 5, 5]])
    def test_homogenous_groups(self, n_obs_per_group: list, n_neighbors=None):
        """Here the NN are expected to be always of the same type"""

        if n_neighbors is None:
            n_neighbors = min(n_obs_per_group) - 1
        else:
            assert n_neighbors < min(n_obs_per_group)

        feat, labs = get_n_groups_data(n_obs_per_group)

        # fit
        indv = Individuality(n_neighbors=n_neighbors)
        res = indv.predict(feat, labs)
        assert (np.diag(res) == np.ones(len(res))).all()

    @pytest.mark.parametrize("n_obs_per_group", [[1], [2, 2]])
    def test_0_neighbors_exception(self, n_obs_per_group: list, n_neighbors: int = 0):
        with pytest.raises(ValueError):
            feat, labs = get_n_groups_data(n_obs_per_group)

            indv = Individuality(n_neighbors=n_neighbors)
            indv.predict(feat, labs)

    @pytest.mark.parametrize("n_obs_per_feat_group", [[2], [6, 6], [4, 4, 4, 4]])
    def test_two_groups_probability(self, n_obs_per_feat_group: list):
        """
        All probabilities are 0 except for 1 or 2 classes. For the special case ``n_obs_per_feat_group=[2]`` there is
        only one class with probability=1.0. For all other cases two classes should have probabilities > 0.

        We split the `n_obs_per_group` into two further groups of equal size and assign labels. Thus in the features
        space there exist len(n_obs_per_group) groups but in the label space each of the identical observations are
        split in two groups.
        """

        n_obs_per_feat_group = np.asarray(n_obs_per_feat_group)
        if not (n_obs_per_feat_group % 2 == 0).all():
            raise ValueError(f"Each group must be a multiple of two")
        if not np.equal(n_obs_per_feat_group, n_obs_per_feat_group).all():
            raise ValueError(
                """
                Not all groups have the same size. If the groups have not the same size the selected.
                If the groups are not the same size the selected neighbours can be ambiguous since all observations in 
                the `n_obs_per_feat_group` have the same feature vector.
                """
            )

        feat = get_n_groups_feat(n_obs_per_feat_group)
        labs = get_n_groups_label((np.array(n_obs_per_feat_group) // 2).repeat(2))
        n_neighbors = n_obs_per_feat_group.min() - 1

        expected_val_on_diag = np.repeat(
            (n_obs_per_feat_group // 2 - 1) / n_neighbors, 2
        )

        indv = Individuality(n_neighbors=n_neighbors)
        res = indv.predict(feat, labs)

        assert np.isclose(np.diag(res), expected_val_on_diag).all()
        assert (res.values == res.values.T).all()
        assert np.isclose(res.values.sum(1), 1).all()

    def test_all_different_groups_probability(self):
        pass

    @pytest.mark.parametrize(
        "n_obs_per_group,n_neighbors", [([2], 1), ([4, 4], 2), ([2, 4, 8], 5)]
    )
    def test_non_numeric_labels(self, n_obs_per_group: list, n_neighbors: int):
        feat, labs = get_n_groups_data(n_obs_per_group)

        n_labs = len(np.unique(labs))
        num2str = {
            i: string.ascii_lowercase[i] for i, j in zip(np.unique(labs), range(n_labs))
        }
        str_labs = [num2str[i] for i in labs]

        ad = AnnData(X=feat, obs=pd.DataFrame({"labels": str_labs}))

        # fit
        indv = Individuality(n_neighbors=n_neighbors)
        res = indv.predict(ad, ad.obs.labels)

        assert (res.index == np.unique(str_labs)).all()
        assert (res.columns == np.unique(str_labs)).all()

    @pytest.mark.parametrize("n_obs_per_group", [[2], [4, 4], [2, 4, 8]])
    def test_provide_graph(self, n_obs_per_group: list):
        labs = get_n_groups_feat(n_obs_per_group)
        np.random.random((len(labs), len(labs)))
