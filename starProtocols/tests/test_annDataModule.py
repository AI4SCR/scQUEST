# %%
import pytest

from starProtocols.preprocessing import StandardScale
from starProtocols.data import AnnDataModule, AnnDatasetAE, AnnDatasetClf
from .utils import dummy_annData

from itertools import product


# %%

@pytest.fixture
def dummy_ad():
    return dummy_annData()


ad_ds = [AnnDatasetClf, AnnDatasetAE]
params = product(['fit', None], ad_ds)


@pytest.mark.parametrize('setup,ad_dataset_cls', params)
def test_setup_fit_None_annDataModule(dummy_ad, setup, ad_dataset_cls):
    dm = AnnDataModule(dummy_ad, target='y_id', ad_dataset_cls=ad_dataset_cls)
    dm.setup(setup)
    assert dm.train is not None
    assert dm.fit is not None
    assert dm.val is not None
    assert dm.test is not None


@pytest.mark.parametrize('ad_dataset_cls', ad_ds)
def test_setup_test_annDataModule(dummy_ad, ad_dataset_cls):
    dm = AnnDataModule(dummy_ad, target='y_id', ad_dataset_cls=ad_dataset_cls)
    dm.setup('test')
    assert dm.train is not None
    assert dm.fit is None
    assert dm.val is None
    assert dm.test is not None


processor_params = product(ad_ds, [StandardScale])


@pytest.mark.parametrize('ad_dataset_cls,processors', processor_params)
def test_processors(dummy_ad, ad_dataset_cls, processors):
    AnnDataModule(dummy_ad, target='y_id',
                  ad_dataset_cls=ad_dataset_cls,
                  preprocessing=processors)
