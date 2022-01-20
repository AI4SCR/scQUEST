# %%
from itertools import tee
from typing import Optional, Iterable

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset


# NOTE: pairwise is distributed with `itertool` in python>=3.10
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# %%

class Classifier(nn.Module):

    def __init__(self, n_in: int, hidden: Iterable[int], n_out: int,
                 bias=True,
                 activation=torch.relu,
                 activation_last=torch.sigmoid):
        super(Classifier, self).__init__()

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


class LitClassifier(pl.LightningModule):

    def __init__(self, model: nn.Module,
                 loss_fn=nn.MSELoss(),
                 learning_rate=1e-3):
        super(LitClassifier, self).__init__()
        self.model = model
        self.loss = loss_fn
        self.learning_rate = learning_rate

    def forward(self, x) -> int:
        return torch.argmax(self.model(x), axis=1).detach()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        return loss

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DummyData(Dataset):

    def __init__(self):
        super(DummyData, self).__init__()
        self.data, self.targets = self.generate_data()

    def __getitem__(self, item):
        return self.data[item,], self.targets[item]

    def __len__(self):
        return self.data.shape[0]

    def generate_data(self):
        n1 = n2 = 1000
        cls1 = torch.randn((1, 5)).repeat((n1, 1))
        cls2 = torch.randn((1, 5)).repeat((n2, 1))
        data = torch.vstack((cls1, cls2))

        y = torch.hstack((torch.tensor(0).repeat(n1), torch.tensor(1).repeat(n2)))
        targets = F.one_hot(y)

        return data.float(), targets.float()


class LitDummyData(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, test_size: float = .2, validation_size: float = .2):
        super(LitDummyData).__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.data = DummyData()

    # how to download, tokenize, etc…
    def prepare_data(self) -> None:
        pass

    # how to split, define dataset, etc…
    def setup(self, stage: Optional[str] = None):
        data = self.data
        n_test = int(len(data) * self.test_size)
        n_train = len(data) - n_test

        if stage in ('fit', None):
            train, _ = random_split(data, [n_train, n_test], generator=torch.Generator().manual_seed(42))

            n_val = int(len(train) * self.validation_size)
            n_fit = len(train) - n_val

            self.fit, self.val = random_split(train, [n_fit, n_val], generator=torch.Generator().manual_seed(42))

        elif stage == 'test':
            _, self.test = random_split(data, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.fit, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        pass

    # Used to clean-up when the run is finished
    def teardown(self, stage: Optional[str] = None):
        pass

    def __getitem__(self, item):
        return self.data[item]
