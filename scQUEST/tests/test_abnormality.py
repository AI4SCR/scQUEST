# %%
import pytest
from .utils import dummy_annData

from scQUEST.abnormality import DefaultAE, Abnormality, AbnormalityLitModule
from scQUEST import DEFAULT_N_FEATURES

import numpy as np
from anndata import AnnData


# %%

@pytest.fixture
def dummy_ad():
    return dummy_annData(n_feat=DEFAULT_N_FEATURES)


@pytest.fixture
def dummy_ad_ones(n_obs=1000, n_feat=27):
    return AnnData(X=np.ones((n_obs, n_feat)))


@pytest.mark.parametrize('n_in,hidden', [(8, []), (8, [4, ]), (8, [4, 4])])
def test_model_load(n_in, hidden):
    model = DefaultAE(n_in, hidden)


def test_litModule_load():
    pass


def test_reproducible_model_init():
    m1 = DefaultAE(5, [3, 3])
    m2 = DefaultAE(5, [3, 3])

    M = [m1, m2]
    P = [m.parameters() for m in M]
    for w1, w2 in zip(*P):
        assert (w1 == w2).all()


def test_default(dummy_ad):
    est = Abnormality()
    est.fit(dummy_ad, max_epochs=5)


def test_model_is_fitting(dummy_ad_ones):
    est = Abnormality()
    est.fit(dummy_ad_ones, early_stopping=False, max_epochs=500)
    est.predict(dummy_ad_ones)
    assert np.isclose(dummy_ad_ones.layers['abnormality'], 0, rtol=0, atol=0.01).all()
