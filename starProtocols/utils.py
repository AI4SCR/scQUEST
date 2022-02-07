from pandas.api.types import CategoricalDtype
from itertools import tee

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Iterable, Optional, List, Union
from .preprocessing import Preprocessor

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader

from scipy.sparse import issparse
from anndata import AnnData

from .data import AnnDataModule

# %%
DEFAULT_MARKERS = {'AKT',
                   'AR',
                   'BCL2',
                   'CA9',
                   'CD24',
                   'CD44',
                   'CD49f',
                   'ECadherin',
                   'EGFR',
                   'ERa',
                   'EZH2',
                   'EpCAM',
                   'HER2',
                   'HLADR',
                   'K14',
                   'K5',
                   'K7',
                   'K8K18',
                   'PRB',
                   'PTEN',
                   'SMA',
                   'Survivin',
                   'Vimentin',
                   'cMET',
                   'cMYC',
                   'p53',
                   'panK'}
DEFAULT_N_FEATURES = len(DEFAULT_MARKERS)


# %%

# NOTE: pairwise is distributed with `itertool` in python>=3.10
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def isCategorical(x):
    return isinstance(x, CategoricalDtype)


class LitModule(pl.LightningModule):

    def __init__(self, model: nn.Module,
                 loss_fn,
                 metrics: Iterable[torchmetrics.Metric],
                 learning_rate=1e-3):
        super(LitModule, self).__init__()

        self.model = model
        self.loss = loss_fn

        # we create for each metric a own attribute to use be able to automatic logging with lightning
        self.metric_attrs = []
        for metric in metrics:
            attr = f'metric_{metric._get_name()}'.lower()
            setattr(self, attr, metric)
            self.metric_attrs.append(attr)

        self.learning_rate = learning_rate

    def forward(self, x) -> int:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log('fit_loss', loss.detach())
        self.log_metrics('train', y, yhat)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log('val_loss', loss.detach())
        self.log_metrics('val', y, yhat)
        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log('test_loss', loss.detach())
        self.log_metrics('test', y, yhat)

        return loss

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_metrics(self, step, y, yhat):
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        true_cls, pred_cls = y.argmax(axis=1), yhat.argmax(axis=1)

        for metric in self.metric_attrs:
            m = getattr(self, metric)
            m(pred_cls, true_cls)
            self.log(f'{step}_{metric}', m)


class Estimator:

    def __init__(self, model: Optional[nn.Module] = None,
                 loss_fn: Optional = None,
                 metrics: Optional = None,
                 seed: Optional[int] = None,
                 ):
        """Base estimator class

        Args:
            model: Model used to train estimator :class:`.torch.Module` or :class:`.pytorch_lightning.Module`
            loss_fn: Loss function used for optimization
            metrics: Metrics tracked during test time
            seed: Seed for model weight initialisation
        """
        self.ad = None
        self.target = None
        self.datamodule = None
        self.trainer = None

        self.loss_fn = loss_fn if loss_fn else self._default_loss()
        self.metrics = metrics if metrics else self._default_metric()
        self.seed = seed if seed else 41

        LM = self._default_litModule()

        if model is None:
            self.model = LM(model=self._default_model(seed=self.seed),
                            loss_fn=self.loss_fn,
                            metrics=self.metrics)
        else:
            if isinstance(model, pl.LightningModule):
                self.model = model
            elif isinstance(model, nn.Module):
                self.model = LM(model=model,
                                loss_fn=self.loss_fn,
                                metrics=self.metrics)
            else:
                raise TypeError(f'Model should be of type LightningModule or nn.Module not {type(model)}')

    def fit(self, ad: Optional[AnnData] = None, target: Optional[str] = None,
            layer: Optional[str] = None,
            datamodule: Optional[pl.LightningDataModule] = None,
            preprocessing: Optional[List[Preprocessor]] = None,
            early_stopping: Union[bool, EarlyStopping] = True,
            max_epochs: int = 100,
            callbacks: list = None,
            seed: Optional[int] = None) -> None:
        """Fit the estimator.

        Args:
            ad: AnnData object to fit
            target: column in AnnData.obs that should be used as target variable
            layer: layer in `ad.layers` to use instead of ad.X
            datamodule: pytorch lightning data module
            preprocessing: list of processors that should be applied to the dataset
            early_stopping: configured :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` class
            max_epochs: maximum epochs for which the model is trained
            callbacks: additional `pytorch_lightning callbacks`
            seed: Seed for data split

        Returns:
            None
        """

        raise NotImplementedError()

    def predict(self, ad: AnnData, layer: Optional[str] = None, inplace=True) -> AnnData:
        """

        Args:
            ad: AnnData object to fit
            layer: AnnData.X layer to use for prediction
            inplace: whether to manipulate the AnnData object inplace or return a copy

        Returns:
            None or AnnData depending on `inplace`.
        """
        raise NotImplementedError()

    def _fit(self, ad: Optional[AnnData] = None, target: Optional[str] = None,
             layer: Optional[str] = None,
             datamodule: Optional[pl.LightningDataModule] = None,
             preprocessing: Optional[List[Preprocessor]] = None,
             early_stopping: Union[bool, EarlyStopping] = True,
             max_epochs: int = 100,
             callbacks: list = None,
             seed: Optional[int] = None):
        callbacks = [] if callbacks is None else callbacks
        self.target = 'target' if target is None else target

        if datamodule is None:
            self.datamodule = AnnDataModule(ad=ad, target=target, layer=layer,
                                            ad_dataset_cls=self._configure_anndata_class(),
                                            preprocessing=preprocessing,
                                            seed=seed)
        else:
            self.datamodule = datamodule

        if early_stopping:
            if isinstance(early_stopping, EarlyStopping):
                callbacks.append(early_stopping)
            else:
                callbacks.append(EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=1e-3,
                                               patience=10))

        self.trainer = pl.Trainer(logger=False,
                                  enable_checkpointing=False,
                                  max_epochs=max_epochs,
                                  callbacks=callbacks,
                                  reload_dataloaders_every_epoch=False)
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def _default_model(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError()

    def _configure_anndata_class(self) -> nn.Module:
        raise NotImplementedError()

    def _default_loss(self):
        raise NotImplementedError()

    def _default_metric(self):
        raise NotImplementedError()

    def _default_litModule(self):
        raise NotImplementedError()

    def _predict(self, ad: AnnData, layer: Optional[str] = None, inplace: bool = True):
        self.ad = ad if inplace else ad.copy()

        X = ad.X if layer is None else ad.layers[layer]
        X = X.A if issparse(X) else X
        X = torch.tensor(X).float()

        self._predict_step(X)

        if not inplace:
            return ad

    def _predict_step(self, X):
        raise NotImplementedError()
