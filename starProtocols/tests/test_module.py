"""Unit testing for module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

from itertools import product

import numpy as np
import pytest

from starProtocols import Individuality


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

    @pytest.mark.parametrize('n_obs_per_group', [[2], [2, 2], [5, 5], [3, 5, 5]])
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

    @pytest.mark.parametrize('n_obs_per_group', [[1], [2, 2]])
    def test_0_neighbors_exception(self, n_obs_per_group: list, n_neighbors: int = 0):
        with pytest.raises(ValueError):
            feat, labs = get_n_groups_data(n_obs_per_group)

            indv = Individuality(n_neighbors=n_neighbors)
            indv.predict(feat, labs)

    def test_all_40_60_probability(self, n_obs_per_group: list = [6, 6]):
        n_obs_per_group = np.asarray(n_obs_per_group)
        if not (n_obs_per_group % 2 == 0).all():
            raise ValueError(f'Each group must be a multiple of two')

        feat = get_n_groups_feat(n_obs_per_group)
        labs = get_n_groups_label((np.array(n_obs_per_group) // 2).repeat(2))
        n_neighbors = n_obs_per_group.min() - 1

        indv = Individuality(n_neighbors=n_neighbors)
        res = indv.predict(feat, labs)
        
        assert np.isclose(np.diag(res), np.ones(len(res)) * 0.4).all()
        assert (res.values == res.values.T).all()
        assert np.isclose(res.values.sum(1), 1).all()

    def test_all_10_probability(self):
        pass

    def test_no_neighbors(self):
        pass

    def test_non_numeric_labels(self):
        pass

# %%
# n_obs_per_group: list = [1]
# n_groups: int = 2
# n_neighbors = min(n_obs_per_group) - 1
# X, labs = get_n_groups_data(n_obs_per_group)
# indv = Individuality(n_neighbors=n_neighbors)
# indv.predict(X, labs)
