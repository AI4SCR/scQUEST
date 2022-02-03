# %%
import torch
import torch.nn as nn

from typing import Iterable, Optional, Union, List
from anndata import AnnData
import pytorch_lightning as pl
from .preprocessing import Preprocessor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .utils import pairwise, Estimator, LitModule
from .data import AnnDatasetAE
from .utils import DEFAULT_N_FEATURES

import torchmetrics
import numpy as np


# %%

class DefaultAE(nn.Module):
    """
    Default AE as implemented in [Wagner2019]_
    """

    def __init__(self, n_in: int = DEFAULT_N_FEATURES, hidden: Iterable[int] = (10, 2, 10),
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
    """Estimator to quantify the abnormality of a cell's expression profile.

    Args:
        model: Model used to train estimator :class:`.torch.Module` or :class:`.pytorch_lightning.Module`
        loss_fn: Loss function used for optimization
        metrics: Metrics tracked during test time

    """

    def __init__(self, *args, **kwargs):
        super(Abnormality, self).__init__(*args, **kwargs)

    def fit(self, ad: Optional[AnnData] = None,
            layer: Optional[str] = None,
            datamodule: Optional[pl.LightningDataModule] = None,
            preprocessing: Optional[List[Preprocessor]] = None,
            early_stopping: Union[bool, EarlyStopping] = True,
            max_epochs: int = 100,
            callbacks: list = None,
            **kwargs) -> None:
        """Fit abnormality estimator.

        Args:
            ad: AnnData object to fit
            layer: layer in `ad.layers` to use instead of ad.X
            datamodule: pytorch lightning data module with custom configurations of train, val and test splits
            preprocessing: list of processors (:class:`~starProtocols.preprocessing.Preprocessor`) that should be applied to the dataset
            early_stopping: configured :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` class
            max_epochs: maximum epochs for which the model is trained
            callbacks: additional `pytorch_lightning callbacks`

        Returns:
            None
        """
        self._fit(ad=ad, layer=layer, datamodule=datamodule, preprocessing=preprocessing,
                  early_stopping=early_stopping, max_epochs=max_epochs, callbacks=callbacks)

    def predict(self, ad: AnnData, layer: Optional[str] = None, inplace=True) -> AnnData:
        """Predict abnormality of each cell.

        Args:
            ad: AnnData object to fit
            layer: `AnnData.X` layer to use for prediction
            inplace: whether to manipulate the AnnData object inplace or return a copy

        Returns:
            None or AnnData depending on `inplace`.
        """
        self._predict(ad, layer, inplace)

    def _predict_step(self, X):
        self.ad.layers['abnormality'] = self.model(X).detach().numpy()

    @staticmethod
    def aggregate(ad, agg_fun=np.mean, key='abnormality', layer='abnormality'):
        res = agg_fun(ad.layers[layer], axis=1)
        ad.obs[key] = res

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
