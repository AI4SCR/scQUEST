"""Module initialization."""

__version__ = "0.1.0"

from .abnormality import Abnormality
from .individuality import Individuality
from .classifier import EpithelialClassifier
from .preprocessing import StandardScale
from . import dataset

from pathlib import Path

ROOT = Path('~/.starProtocols').expanduser()
ROOT.mkdir(exist_ok=True)
