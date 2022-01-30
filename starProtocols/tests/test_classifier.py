# %%
import pytest
from starProtocols.classifier import DefaultCLF, EpithelialClassifier
from .utils import dummy_annData


# %%

@pytest.fixture
def dummy_ad():
    return dummy_annData(n_feat=25)


@pytest.mark.parametrize('n_in,hidden', [(8, []), (8, [4, ]), (8, [4, 4])])
def test_model_load(n_in, hidden):
    model = DefaultCLF(n_in, hidden)


def test_litModule_load():
    pass


def test_reproducible_model_init():
    m1 = DefaultCLF(5, [3, 3])
    m2 = DefaultCLF(5, [3, 3])

    M = [m1, m2]
    P = [m.parameters() for m in M]
    for w1, w2 in zip(*P):
        assert (w1 == w2).all()


def test_default(dummy_ad):
    est = EpithelialClassifier()
    est.fit(dummy_ad, target='y_id', max_epochs=5)
