# %%
from typing import Iterable, Optional, Union, List, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from anndata import AnnData

from .data import AnnDatasetAE
from .utils import pairwise, Estimator, LitModule


# %%


class DefaultAE(nn.Module):
    """
    Default AE as implemented in [Wagner2019]_
    """

    def __init__(
        self,
        n_in: int,
        hidden: Iterable[int] = (10, 2, 10),
        bias=True,
        activation=nn.ReLU(),
        activation_last=nn.Sigmoid(),
        seed: Optional[int] = None,
    ):
        super(DefaultAE, self).__init__()

        self.n_in = n_in
        self.hidden = hidden
        self.bias = bias
        self.activation = activation
        self.activation_last = activation_last
        self.seed = seed if seed else 41

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
    """pytorch_lightning Abnormality module"""

    def __init__(self, *args, **kwargs):
        super(AbnormalityLitModule, self).__init__(*args, **kwargs)

    def forward(self, X, mode: Optional[str] = None) -> int:
        """return the reconstruction error"""
        return X - self.model(X)


class Abnormality(Estimator):
    """Estimator to quantify the abnormality of a cell's expression profile. Abnormality is defined as the average
    reconstruction error of the autoencoder trained on a reference (normal) cell population.

    Args:
        n_in: number of feature for estimator
        model: Model used to train estimator :class:`.torch.Module` or :class:`.pytorch_lightning.Module`
        loss_fn: Loss function used for optimization
        metrics: Metrics tracked during test time

    Note:
        The abnormality model (`Abn.model`) predicts the abnormality a.k.a. reconstruction error, i.e. :math:`X-F(X)`
        where :math:`F` is the autoencoder (saved to `ad.layers['abnormality']`). On the other hand, the base torch
        model (`Abn.model.model`) predicts the reconstruction, i.e. :math:`F(X)`.

    """

    def __init__(
        self,
        n_in: Optional[int] = None,
        model: Optional[nn.Module] = None,
        loss_fn: Optional = None,
        metrics: Optional = None,
        seed: Optional[int] = None,
    ):
        super(Abnormality, self).__init__(
            n_in=n_in, model=model, loss_fn=loss_fn, metrics=metrics, seed=seed
        )

    def fit(
        self,
        ad: Optional[AnnData] = None,
        layer: Optional[str] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        max_epochs: int = 100,
        callbacks: list = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Fit abnormality estimator (autoencoder). Given the cell-expression profile given in ad.X or ad.layer[layer], an
        autoencoder is fitted. By default the given data is randomly split 90/10 in training and test set. If you wish to
        customize training provide a datamodule with the given train/validation/test splits.

        Args:
            ad: AnnData object to fit
            layer: layer in `ad.layers` to use instead of ad.X
            datamodule: pytorch lightning data module with custom configurations of train, val and test splits
            preprocessing: list of processors (:class:`~scQUEST.preprocessing.Preprocessor`) that should be applied to the dataset
            early_stopping: configured :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` class
            max_epochs: maximum epochs for which the model is trained
            callbacks: additional `pytorch_lightning callbacks`
            seed: Seed for data split

        Returns:
            None
        """
        self._fit(
            ad=ad,
            layer=layer,
            datamodule=datamodule,
            max_epochs=max_epochs,
            callbacks=callbacks,
            seed=seed,
            **kwargs,
        )

    def predict(
        self, ad: AnnData, layer: Optional[str] = None, inplace=True
    ) -> AnnData:
        """Predict abnormality of each cell-feature as the difference between target and reconstruction (y-pred).

        Args:
            ad: AnnData object to fit
            layer: `AnnData.X` layer to use for prediction
            inplace: whether to manipulate the AnnData object inplace or return a copy

        Returns:
            None or AnnData depending on `inplace`.
        """
        self._predict(ad, layer, inplace)

    def _predict_step(self, X):
        self.ad.layers["abnormality"] = self.model(X).detach().numpy()

    @staticmethod
    def aggregate(
        ad,
        agg_fun: Union[str, Callable] = "mse",
        key="abnormality",
        layer="abnormality",
    ):
        """Aggregate the high-dimensional (number of features) reconstruction error of each cell.

        :param ad: AnnData object
        :param agg_fun: `mse` or function used to aggregate the observed reconstruciton errors
        :param key: key under which the results should be stored in ad.obs
        :param layer: layer in X used to compute the aggregation
        """
        if agg_fun == "mse":
            res = (ad.layers[layer] ** 2).mean(axis=1)
        else:
            res = agg_fun(ad.layers[layer], axis=1)
        ad.obs[key] = res

    def _default_model(self, *args, **kwargs) -> nn.Module:
        return DefaultAE(*args, **kwargs)

    def _default_loss(self):
        return nn.MSELoss()

    def _default_metric(self):
        return (torchmetrics.MeanSquaredError(),)

    def _default_litModule(self):
        return AbnormalityLitModule

    def _configure_anndata_class(self):
        return AnnDatasetAE
