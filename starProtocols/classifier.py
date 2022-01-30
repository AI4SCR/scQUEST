# %%
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchmetrics

from .utils import pairwise, Estimator, LitModule
from .data import AnnDatasetClf

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader


# %%

class DefaultCLF(nn.Module):

    def __init__(self, n_in: int = 25, hidden: Iterable[int] = (20,), n_out: int = 2,
                 bias=True,
                 activation=nn.ReLU(),
                 activation_last=nn.Softmax(dim=1),
                 seed: int = 0):
        super(DefaultCLF, self).__init__()

        self.n_in = n_in
        self.hidden = hidden
        self.n_out = n_out
        self.bias = bias
        self.activation = activation
        self.activation_last = activation_last
        self.seed = seed

        # fix seeds
        torch.manual_seed(self.seed)

        layers = []
        for i, j in pairwise([n_in, *hidden, n_out]):
            layers.append(nn.Linear(i, j, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.activation_last(self.layers[-1](x))


class ClfLitModule(LitModule):

    def __init__(self, *args, **kwargs):
        super(ClfLitModule, self).__init__(*args, **kwargs)

    def forward(self, x) -> int:
        return self.model(x).argmax(axis=1)


class EpithelialClassifier(Estimator):

    def __init__(self, *args, **kwargs):
        super(EpithelialClassifier, self).__init__(*args, **kwargs)

    def _predict(self, X):
        yhat = self.model(X)
        self.ad.obs[self.target] = yhat.numpy()

    def _default_model(self) -> nn.Module:
        return DefaultCLF()

    def _default_loss(self):
        return nn.CrossEntropyLoss()

    def _default_metric(self):
        return (torchmetrics.Accuracy(), torchmetrics.Precision())

    def _default_litModule(self):
        return ClfLitModule

    def _configure_anndata_class(self):
        return AnnDatasetClf
