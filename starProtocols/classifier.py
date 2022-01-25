# %%
from functools import reduce
from itertools import tee
from typing import Optional, Iterable, List, Union

import torchmetrics
import pytorch_lightning as pl
import torch
from anndata import AnnData
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import random_split, DataLoader

from .data import AnnDataset
from .preprocessing import Preprocessor

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader


# NOTE: pairwise is distributed with `itertool` in python>=3.10
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# %% data

class AnnDataModule(pl.LightningDataModule):

    def __init__(self, ad: AnnData, target: str, layer: Optional[str] = None,
                 preprocessing: Optional[List[Preprocessor]] = None,
                 test_size: float = 0.25, validation_size: float = 1 / 3,
                 batch_size: int = 32, seed: int = 42):
        super(AnnDataModule, self).__init__()

        self.ad = ad
        self.target = target
        self.layer = layer
        self.preprocessing = preprocessing
        self.test_size = test_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.seed = seed

        self.dataset = AnnDataset(ad, target=target, layer=layer)

        # datasets
        self.train = None
        self.fit = None
        self.val = None
        self.test = None

    # TODO: How should we preprocess the data? Preprocessing should be part of the DataModule to keep everything related to data together
    #   however, in its current form we have to implement the Processor for torch Datasets and Subsets.
    def setup(self, stage: Optional[str] = None) -> None:
        dataset = self.dataset
        n_test = int(len(dataset) * self.test_size)
        n_train = len(dataset) - n_test

        self.train, self.test = random_split(dataset, [n_train, n_test],
                                             generator=torch.Generator().manual_seed(self.seed))
        self.fit_preprocessors()  # TODO: Should we fit the preprocessors on the whole train or on fit,val individually?

        if stage in ('fit', None):
            train = self.train
            if self.preprocessing:
                for pp in self.preprocessing:
                    self.train.dataset.data = pp.transform(self.train)

            n_val = int(len(train) * self.validation_size)
            n_fit = len(train) - n_val

            self.fit, self.val = random_split(train, [n_fit, n_val], generator=torch.Generator().manual_seed(self.seed))

        if stage in ('test', None):
            if self.preprocessing:
                for pp in self.preprocessing:
                    self.test.dataset.data = pp.transform(self.test)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.fit, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size)

    def fit_preprocessors(self):
        if self.preprocessing:
            for pp in self.preprocessing:
                pp.fit(self.train)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


# %%

class DefaultClassifier(nn.Module):

    def __init__(self, n_in: int, hidden: Iterable[int] = (20,), n_out: int = 2,
                 bias=True,
                 activation=nn.ReLU(),
                 activation_last=nn.Softmax(dim=1)):
        super(DefaultClassifier, self).__init__()

        self.n_in = n_in
        self.hidden = hidden
        self.n_out = n_out
        self.bias = bias
        self.activation = activation
        self.activation_last = activation_last

        layers = []
        for i, j in pairwise([n_in, *hidden, n_out]):
            layers.append(nn.Linear(i, j, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.activation_last(self.layers[-1](x))


class LitModule(pl.LightningModule):

    def __init__(self, model: nn.Module,
                 loss_fn=nn.CrossEntropyLoss(),
                 metrics: Iterable[torchmetrics.Metric] = (torchmetrics.Accuracy(), torchmetrics.Precision()),
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
        return torch.argmax(self.model(x), axis=1).detach()

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


class EpithelialClassifier:

    def __init__(self, n_in: int = 5, model: Optional[nn.Module] = None):

        self.n_in = n_in
        self.ad = None
        self.target = None
        self.datamodule = None
        self.trainer = None

        if model is None:
            self.model = LitModule(model=DefaultClassifier(n_in=n_in))
        else:
            if isinstance(model, pl.LightningModule):
                self.model = model
            elif isinstance(model, nn.Module):
                self.model = LitModule(model=model)
            else:
                raise TypeError(f'Model should be of type LightningModule or nn.Module not {type(model)}')

    def fit(self, ad: Optional[AnnData] = None, target: Optional[str] = None,
            datamodule: Optional[pl.LightningDataModule] = None,
            preprocessing: Optional[List[Preprocessor]] = None,
            early_stopping: Union[bool, EarlyStopping] = True,
            max_epochs: int = 100,
            callbacks: list = None):

        callbacks = [] if callbacks is None else callbacks
        self.target = 'target' if target is None else target

        if datamodule is None:
            self.datamodule = AnnDataModule(ad, target, preprocessing=preprocessing)
        else:
            self.datamodule = datamodule

        if early_stopping:
            if isinstance(early_stopping, EarlyStopping):
                callbacks.append(early_stopping)
            else:
                callbacks.append(EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=1e-3,
                                               patience=0))

        self.trainer = pl.Trainer(logger=False,
                                  enable_checkpointing=False,
                                  max_epochs=max_epochs,
                                  callbacks=callbacks,
                                  reload_dataloaders_every_epoch=False)
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def predict(self, ad: AnnData, layer: Optional[str] = None, inplace=True):
        ad = ad if inplace else ad.copy()

        X = ad.X if layer is None else ad.layers[layer]
        X = X.A if issparse(X) else X
        X = torch.tensor(X).float()
        yhat = self.model(X)

        ad.obs[self.target] = yhat.numpy()

        if not inplace:
            return ad
