"""Unit testing for module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import numpy as np
from itertools import product

from starProtocols import Individuality


def get_two_groups_feat(n_obs: int = 10, n_feat: int = 10):
    feat = np.zeros((n_obs, n_feat))
    feat[:n_obs // 2] = 1
    return feat


def get_two_groups_label(n_obs: int = 10):
    labs = np.zeros(n_obs)
    labs[:n_obs // 2] = 1
    return labs


def get_two_groups_data(n_obs: int = 10, n_feat: int = 10):
    return get_two_groups_feat(n_obs, n_feat), get_two_groups_label(n_obs)


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

    def test_homogenous_groups(self, n_obs_per_group: int = 5, n_groups: int = 2, n_neighbors=None):
        """Here the NN are expected to be always of the same type"""

        if n_neighbors is None:
            n_neighbors = n_obs_per_group - 1
        else:
            assert n_neighbors < n_obs_per_group

        # get data
        n_obs = (n_obs_per_group) * n_groups
        feat, labs = get_n_groups_data(n_obs, n_groups)

        # fit
        indv = Individuality(n_neighbors=n_neighbors)
        res = indv._predict(feat, labs)

    def test_all_50_propability(self):
        pass

    def test_all_10_propability(self):
        pass

    def test_no_neighbors(self):
        pass

    def test_non_numeric_labels(self):
        pass


# %%
n_obs_per_group: int = 5
n_groups: int = 2
n_neighbors = n_obs_per_group - 1
X, labs = get_n_groups_data(n_obs_per_group * n_groups, n_groups)
indv = Individuality(n_neighbors=n_neighbors)
indv.predict(X, labs)
