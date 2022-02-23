# %%
from typing import Iterable, Optional, Union, List
from anndata import AnnData
import pytorch_lightning as pl
from .preprocessing import Preprocessor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchmetrics

from ._utils import pairwise, Estimator, LitModule
from ._data import AnnDatasetClf

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader


# %%

class DefaultCLF(nn.Module):

    def __init__(self, n_in: int, hidden: Iterable[int] = (20,), n_out: int = 2,
                 bias=True,
                 activation=nn.ReLU(),
                 activation_last=nn.Softmax(dim=1),
                 seed: Optional[int] = None):
        super(DefaultCLF, self).__init__()

        self.n_in = n_in
        self.hidden = hidden
        self.n_out = n_out
        self.bias = bias
        self.activation = activation
        self.activation_last = activation_last
        self.seed = seed if seed else 41

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
    """Classifier to classify a cell as epithelial or non-epithelial cell.

    Args:
        n_in: number of features
        model: Model used to train estimator :class:`.torch.Module` or :class:`.pytorch_lightning.Module`
        loss_fn: Loss function used for optimization
        metrics: Metrics tracked during test time
    """

    def __init__(self, n_in: Optional[int],
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional = None,
                 metrics: Optional = None,
                 seed: Optional[int] = None):
        super(EpithelialClassifier, self).__init__(n_in=n_in, model=model, loss_fn=loss_fn, metrics=metrics, seed=seed)

    def fit(self, ad: Optional[AnnData] = None, target: Optional[str] = None,
            layer: Optional[str] = None,
            datamodule: Optional[pl.LightningDataModule] = None,
            preprocessing: Optional[List[Preprocessor]] = None,
            early_stopping: Union[bool, EarlyStopping] = True,
            max_epochs: int = 100,
            callbacks: list = None,
            seed: Optional[int] = None,
            ) -> None:
        """Fit the estimator.

        Args:
            ad: AnnData object to fit
            target: column in AnnData.obs that should be used as target variable
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
        self._fit(ad=ad, target=target, layer=layer, datamodule=datamodule, preprocessing=preprocessing,
                  early_stopping=early_stopping, max_epochs=max_epochs, callbacks=callbacks, seed=seed)

    def predict(self, ad: AnnData, layer: Optional[str] = None, inplace=True) -> AnnData:
        """Predict phenotype class.

        Args:
            ad: AnnData object to fit
            layer: `AnnData.X` layer to use for prediction
            inplace: whether to manipulate the AnnData object inplace or return a copy

        Returns:
            None or AnnData depending on `inplace`.
        """
        self._predict(ad, layer, inplace)

    def _predict_step(self, X):
        yhat = self.model(X)
        self.ad.obs[f'clf_{self.target}'] = yhat.numpy()

    def _default_model(self, *args, **kwargs) -> nn.Module:
        """Default model if not provided"""
        return DefaultCLF(*args, **kwargs)

    def _default_loss(self):
        """Default loss if not provided"""
        return nn.CrossEntropyLoss()

    def _default_metric(self):
        """Default metrics if not provided"""
        return (torchmetrics.Accuracy(), torchmetrics.Precision())

    def _default_litModule(self):
        """Lightning module architecture for classifier"""
        return ClfLitModule

    def _configure_anndata_class(self):
        """Dataset class for estimator"""
        return AnnDatasetClf