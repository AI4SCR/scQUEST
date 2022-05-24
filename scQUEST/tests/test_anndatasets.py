# %%
import pytest

from scQUEST.data import AnnDatasetAE, AnnDatasetClf
from ._utils import dummy_annData


# %%


@pytest.fixture
def dummy_ad():
    ad = dummy_annData()
    return ad


@pytest.fixture
def fruit_bowl():
    return [1, 2]


def test_clf_dataloading(fruit_bowl, dummy_ad):
    dsCLF = AnnDatasetClf(dummy_ad, target="y_id")
    assert len(dsCLF[0]) == 2


def test_ae_dataloading(dummy_ad):
    dsAE = AnnDatasetAE(dummy_ad)
    X, X = dsAE[0]
    assert len(dsAE[0]) == 2
    assert (X == X).all()
