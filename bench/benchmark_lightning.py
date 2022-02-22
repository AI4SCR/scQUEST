# %%
import torch
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import os
import scQUEST as sp
from scQUEST.data import DS

from typing import Optional

from pathlib import Path

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

p = Path('//bench/benchmark.out').expanduser()
# %%
ad = sp.dataset.breastCancerAtlasRaw()

x_train, _ = train_test_split(ad.X[ad.obs.tissue_type == 'N'], test_size=.3)
x_train = x_train[:int(1.5e6), :50]
d1 = x_train.shape[1]

model = sys.argv[1]
setting = sys.argv[2]


# %%

def get_torch_model(d1):
    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(d1, 10), nn.ReLU(),
                                       nn.Linear(10, 2), nn.ReLU(),
                                       nn.Linear(2, 10), nn.ReLU(),
                                       nn.Linear(10, d1), nn.Sigmoid())

        def forward(self, x):
            return self.model(x)

    return Model()


def get_lightning_model(d1):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.loss = nn.MSELoss()
            self.dologging = False
            self.learning_rate = 1e-3
            self.model = get_torch_model(d1)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.dologging:
                self.log('test_loss', loss.detach())
                self.log_metrics('train', y, yhat)
            return loss

        def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.dologging:
                self.log('test_loss', loss.detach())
                self.log_metrics('validate', y, yhat)
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

    return Model()


def get_large_torch_model(d1):
    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(d1, 50), nn.ReLU(),
                                       nn.Linear(50, 50), nn.ReLU(),
                                       nn.Linear(50, 50), nn.ReLU(),
                                       nn.Linear(50, 50), nn.ReLU(),
                                       nn.Linear(50, 2), nn.ReLU(),
                                       nn.Linear(2, 50), nn.ReLU(),
                                       nn.Linear(50, 50), nn.ReLU(),
                                       nn.Linear(50, 50), nn.ReLU(),
                                       nn.Linear(50, 50), nn.ReLU(),
                                       nn.Linear(50, d1), nn.Sigmoid())

        def forward(self, x):
            return self.model(x)

    return Model()


def get_large_lightning_model(d1):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.loss = nn.MSELoss()
            self.logging = False
            self.learning_rate = 1e-3
            self.model = get_large_torch_model(d1)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.logging:
                self.log('test_loss', loss.detach())
                self.log_metrics('train', y, yhat)
            return loss

        def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.logging:
                self.log('test_loss', loss.detach())
                self.log_metrics('validate', y, yhat)
            return loss

        def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.logging:
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

    return Model()


def get_keras_model(d1):
    input_data = Input(shape=(d1,))
    encoded1 = Dense(10, activation='relu')(input_data)
    encoded2 = Dense(2, activation='relu')(encoded1)
    decoded1 = Dense(10, activation='relu')(encoded2)
    decoded2 = Dense(d1, activation='sigmoid')(decoded1)
    ae = Model(input_data, decoded2)
    return ae


def get_large_keras_model(d1):
    input_data = Input(shape=(d1,))
    encoded1 = Dense(50, activation='relu')(input_data)
    encoded2 = Dense(50, activation='relu')(encoded1)
    encoded3 = Dense(50, activation='relu')(encoded2)
    encoded4 = Dense(50, activation='relu')(encoded3)
    encoded5 = Dense(2, activation='relu')(encoded4)
    decoded1 = Dense(50, activation='relu')(encoded5)
    decoded2 = Dense(50, activation='relu')(decoded1)
    decoded3 = Dense(50, activation='relu')(decoded2)
    decoded4 = Dense(50, activation='relu')(decoded3)
    decoded4 = Dense(d1, activation='sigmoid')(decoded4)
    ae = Model(input_data, decoded4)
    return ae


# %%

def fit_torch(model, x_train, epochs=10, batch_size=256):
    def train_loop(dataloader, model, loss_fn, optimizer):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    x_train_ds = DS(x_train)
    train_dataloader = DataLoader(x_train_ds, batch_size=batch_size, num_workers=n_cpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        train_loop(train_dataloader, model, nn.MSELoss(), optimizer)


def fit_lightning(model, x_train, epochs=10, batch_size=256):
    x_train_ds = DS(x_train)
    train_dataloader = DataLoader(x_train_ds, batch_size=batch_size, num_workers=n_cpu)

    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_dataloader)


def fit_lightning_noCheckNoLog(model, x_train, epochs=10, batch_size=256):
    x_train_ds = DS(x_train)
    train_dataloader = DataLoader(x_train_ds, batch_size=batch_size, num_workers=n_cpu)

    trainer = pl.Trainer(logger=False, max_epochs=epochs, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_dataloader)


def fit_lightning_AccPrec(model, x_train, epochs=10, batch_size=256):
    x_train_ds = DS(x_train)
    train_dataloader = DataLoader(x_train_ds, batch_size=batch_size, num_workers=n_cpu)

    trainer = pl.Trainer(logger=False, max_epochs=epochs, accelerator='cpu', precision=16, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_dataloader)


def fit_keras(model, x_train, epochs=10, batch_size=256):
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(x_train, x_train,
              epochs=epochs, verbose=0,
              batch_size=batch_size,
              shuffle=True,
              validation_data=None)


# %%
import time

# n_cpu = os.cpu_count()
n_cpu = 0
n = 5
epochs = 3
batch_size = 256
res = []

for i in range(n):
    if model == 'torch':
        m = get_torch_model(d1)
    elif model == 'large_torch':
        m = get_large_torch_model(d1)
    elif model == 'light':
        m = get_lightning_model(d1)
    elif model == 'large_light':
        m = get_large_lightning_model(d1)
    elif model == 'keras':
        m = get_keras_model(d1)
    elif model == 'large_keras':
        m = get_large_keras_model(d1)

    start = time.time()
    if setting == 'torch':
        fit_torch(m, x_train, epochs=epochs, batch_size=batch_size)
    elif setting == 'light_default':
        fit_lightning(m, x_train, epochs=epochs, batch_size=batch_size)
    elif setting == 'light_noCheckNoLog':
        fit_lightning_noCheckNoLog(m, x_train, epochs=epochs, batch_size=batch_size)
    elif setting == 'light_AccPrec':
        fit_lightning_AccPrec(m, x_train, epochs=epochs, batch_size=batch_size)
    elif setting == 'keras':
        fit_keras(m, x_train, epochs=epochs, batch_size=batch_size)

    res.append((time.time() - start) / epochs)

# %%
import numpy as np

with open(p, 'a') as f:
    s = f'{x_train.shape}\t{model}\t{setting}\tmean:{np.mean(res):.2f}, max: {np.max(res):.2f}, min: {np.min(res):.2f}'
    f.write(s)
    f.write('\n')
