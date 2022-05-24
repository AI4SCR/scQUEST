import abc

from pandas.api.types import CategoricalDtype
from itertools import tee

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Iterable, Optional

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader

from scipy.sparse import issparse
from anndata import AnnData

from .data import AnnDataModule

# %%
DEFAULT_MARKER_CLF = [
    "139La_H3K27me3",
    "141Pr_K5",
    "142Nd_PTEN",
    "143Nd_CD44",
    "144Nd_K8K18",
    "145Nd_CD31",
    "146Nd_FAP",
    "147Sm_cMYC",
    "148Nd_SMA",
    "149Sm_CD24",
    "150Nd_CD68",
    "151Eu_HER2",
    "152Sm_AR",
    "153Eu_BCL2",
    "154Sm_p53",
    "155Gd_EpCAM",
    "156Gd_CyclinB1",
    "158Gd_PRB",
    "159Tb_CD49f",
    "160Gd_Survivin",
    "161Dy_EZH2",
    "162Dy_Vimentin",
    "163Dy_cMET",
    "164Dy_AKT",
    "165Ho_ERa",
    "166Er_CA9",
    "167Er_ECadherin",
    "168Er_Ki67",
    "169Tm_EGFR",
    "170Er_K14",
    "171Yb_HLADR",
    "172Yb_clCASP3clPARP1",
    "173Yb_CD3",
    "174Yb_K7",
    "175Lu_panK",
    "176Yb_CD45",
]

DEFAULT_MARKERS = {
    "AKT",
    "AR",
    "BCL2",
    "CA9",
    "CD24",
    "CD44",
    "CD49f",
    "ECadherin",
    "EGFR",
    "ERa",
    "EZH2",
    "EpCAM",
    "HER2",
    "HLADR",
    "K14",
    "K5",
    "K7",
    "K8K18",
    "PRB",
    "PTEN",
    "SMA",
    "Survivin",
    "Vimentin",
    "cMET",
    "cMYC",
    "p53",
    "panK",
}
DEFAULT_N_FEATURES = len(DEFAULT_MARKER_CLF)


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
    """pytorch_module that handles the training of the model"""

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        metrics: Iterable[torchmetrics.Metric],
        learning_rate=1e-3,
    ):
        super(LitModule, self).__init__()

        self.model = model
        self.loss = loss_fn

        # we create for each metric a own attribute to be able to automatic logging with lightning
        self.metric_attrs = []
        for metric in metrics:
            attr = f"metric_{metric._get_name()}".lower()
            setattr(self, attr, metric)
            self.metric_attrs.append(attr)

        self.learning_rate = learning_rate

    def forward(self, x) -> int:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        if batch_idx % 5:
            self.log("fit_loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log("val_loss", loss.detach())
        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log("test_loss", loss.detach())
        yhat = yhat if y.shape == yhat.shape else yhat.argmax(axis=1)
        self.log_metrics("test", y, yhat)
        return loss

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_metrics(self, step, y, yhat):
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        true_cls, pred_cls = y, yhat

        for metric in self.metric_attrs:
            m = getattr(self, metric)
            m(pred_cls, true_cls)
            self.log(f"{step}_{metric}", m)


class Estimator:
    def __init__(
        self,
        n_in: Optional[int] = None,
        model: Optional[nn.Module] = None,
        loss_fn: Optional = None,
        metrics: Optional = None,
        seed: Optional[int] = None,
    ):
        """Base estimator class

        Args:
            n_in: number of feature for estimator
            model: Model used to train estimator :class:`.torch.Module` or :class:`.pytorch_lightning.Module`
            loss_fn: Loss function used for optimization
            metrics: Metrics tracked during test time
            seed: Seed for model weight initialisation
        """
        self.ad = None
        self.target = None
        self.datamodule = None
        self.trainer = None
        self.logger = MyLogger()

        self.n_in = n_in
        self.loss_fn = loss_fn if loss_fn else self._default_loss()
        self.metrics = metrics if metrics else self._default_metric()
        self.seed = seed if seed else 41

        LM = self._default_litModule()

        if model is None:
            self.model = LM(
                model=self._default_model(n_in=n_in, seed=self.seed),
                loss_fn=self.loss_fn,
                metrics=self.metrics,
            )
        else:
            if isinstance(model, pl.LightningModule):
                self.model = model
            elif isinstance(model, nn.Module):
                self.model = LM(model=model, loss_fn=self.loss_fn, metrics=self.metrics)
            else:
                raise TypeError(
                    f"Model should be of type LightningModule or nn.Module not {type(model)}"
                )

    def fit(
        self,
        ad: Optional[AnnData] = None,
        target: Optional[str] = None,
        layer: Optional[str] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        max_epochs: int = 100,
        callbacks: list = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Fit the estimator.

        Args:
            ad: AnnData object to fit
            target: column in AnnData.obs that should be used as target variable
            layer: layer in `ad.layers` to use instead of ad.X
            datamodule: pytorch lightning data module
            max_epochs: maximum epochs for which the model is trained
            callbacks: additional `pytorch_lightning callbacks`
            seed: Seed for data split

        Returns:
            None
        """

        raise NotImplementedError()

    def predict(
        self, ad: AnnData, layer: Optional[str] = None, inplace=True
    ) -> AnnData:
        """

        Args:
            ad: AnnData object to fit
            layer: AnnData.X layer to use for prediction
            inplace: whether to manipulate the AnnData object inplace or return a copy

        Returns:
            None or AnnData depending on `inplace`.
        """
        raise NotImplementedError()

    def _fit(
        self,
        ad: Optional[AnnData] = None,
        target: Optional[str] = None,
        layer: Optional[str] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        max_epochs: int = 100,
        callbacks: list = None,
        seed: Optional[int] = None,
        **kwargs,
    ):

        callbacks = [] if callbacks is None else callbacks
        self.target = "target" if target is None else target

        if datamodule is None:
            self.datamodule = AnnDataModule(
                ad=ad,
                target=target,
                layer=layer,
                ad_dataset_cls=self._configure_anndata_class(),
                seed=seed,
            )
        else:
            self.datamodule = datamodule

        self.trainer = pl.Trainer(
            logger=self.logger,
            enable_checkpointing=False,
            max_epochs=max_epochs,
            callbacks=callbacks,
            **kwargs,
        )
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


from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
import collections


class MyLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.history = collections.defaultdict(list)

    @property
    def name(self):
        return "MyLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if metric_name != "epoch":
                self.history[metric_name].append((step, metric_value))

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
