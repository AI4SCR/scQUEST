"""Module initialization."""

__version__ = "0.1.0"

from .abnormality import Abnormality
from .individuality import Individuality
from .classifier import Classifier
from . import dataset
from .utils import DEFAULT_N_FEATURES, DEFAULT_MARKERS

from pathlib import Path

ROOT = Path("~/.scQUEST").expanduser()
ROOT.mkdir(exist_ok=True)
