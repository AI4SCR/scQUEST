# %%
import torch
import torch.nn as nn

from typing import Iterable
from .utils import pairwise, Estimator, LitModule
from .data import AnnDatasetAE

import torchmetrics


# %%

class DefaultAE(nn.Module):

    def __init__(self, n_in: int = 25, hidden: Iterable[int] = (10, 2, 10),
                 bias=True,
                 activation=nn.ReLU(),
                 activation_last=nn.Sigmoid(),
                 seed: int = 0):
        super(DefaultAE, self).__init__()

        self.n_in = n_in
        self.hidden = hidden
        self.bias = bias
        self.activation = activation
        self.activation_last = activation_last
        self.seed = seed

        # fix seeds
        torch.manual_seed(self.seed)

        layers = []
        for i, j in pairwise([n_in, *hidden, n_in]):
            layers.append(nn.Linear(i, j, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.activation_last(self.layers[-1](x))


class AbnormalityLitModule(LitModule):
    def __init__(self, *args, **kwargs):
        super(AbnormalityLitModule, self).__init__(*args, **kwargs)

    def forward(self, X) -> int:
        return X - self.model(X)


class Abnormality(Estimator):

    def __init__(self, *args, **kwargs):
        super(Abnormality, self).__init__(*args, **kwargs)

    def _predict(self, X):
        return self.model(X)

    def _default_model(self) -> nn.Module:
        return DefaultAE()

    def _default_loss(self):
        return nn.MSELoss()

    def _default_metric(self):
        return (torchmetrics.MeanSquaredError(),)

    def _default_litModule(self):
        return AbnormalityLitModule

    def _configure_anndata_class(self):
        return AnnDatasetAE
