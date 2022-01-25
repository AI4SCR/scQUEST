from pandas.api.types import CategoricalDtype


def isCategorical(x):
    return isinstance(x, CategoricalDtype)
