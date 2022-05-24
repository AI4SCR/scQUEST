import abc
from abc import ABC
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib import request
import anndata

from typing import Union

PathLike = Union[str, Path]

from ..__init__ import ROOT


@dataclass
class DataSet(ABC):
    name: str
    url: str
    force_load: bool = False

    doc_header: str = field(default=None, repr=False)
    path: PathLike = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.path is None:
            object.__setattr__(self, "path", ROOT / self.name)

    @property
    def _extension(self) -> str:
        return ".h5py"

    def __call__(self, path: PathLike = None):
        return self.load(path)

    def load(self, fpath: PathLike = None):
        """Download dataset form url"""
        fpath = str(self.path if fpath is None else fpath)

        if not fpath.endswith(self._extension):
            fpath += self._extension

        if not os.path.isfile(fpath) or self.force_load:
            # download file
            dirname = Path(fpath).parent
            if not dirname.is_dir():
                dirname.mkdir(parents=True, exist_ok=True)

            self._download_progress(Path(fpath), self.url)

        return anndata.read_h5ad(fpath)

    def _download_progress(self, fpath: Path, url):
        from tqdm import tqdm
        from urllib.request import urlopen, Request

        blocksize = 1024 * 8
        blocknum = 0

        try:
            with urlopen(Request(url, headers={"User-agent": "scQUEST-user"})) as rsp:
                total = rsp.info().get("content-length", None)
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    unit_divisor=1024,
                    total=total if total is None else int(total),
                ) as t, fpath.open("wb") as f:
                    block = rsp.read(blocksize)
                    while block:
                        f.write(block)
                        blocknum += 1
                        t.update(len(block))
                        block = rsp.read(blocksize)
        except (KeyboardInterrupt, Exception):
            # Make sure file doesn’t exist half-downloaded
            if fpath.is_file():
                fpath.unlink()
            raise

    def _download(self, fpath: Path, url) -> None:
        try:
            path, rsp = request.urlretrieve(url, fpath)
        except (KeyboardInterrupt, Exception):
            # Make sure file doesn’t exist half-downloaded
            if path.is_file():
                path.unlink()
            raise
