from abc import ABC, abstractmethod
from pandas.api.types import CategoricalDtype

class Predictor(ABC):

    @abstractmethod
    def fit(self, x, y = None):
        pass

    @abstractmethod
    def predict(self, x, y = None):
        pass

    def fit_predict(self, x, y):
        self.fit(x,y)
        return self.predict(x,y)

def isCategorical(x):
    return isinstance(x, CategoricalDtype)
