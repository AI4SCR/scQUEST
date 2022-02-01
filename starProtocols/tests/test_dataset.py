import starProtocols
import tempfile
from pathlib import Path


def test_download():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'test.h5py'
        ad = starProtocols.dataset.breastCancerAtlas(path=p)


def test_load_from_disk():
    ad = starProtocols.dataset.breastCancerAtlas()
