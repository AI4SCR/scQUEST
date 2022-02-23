# %%
import torch
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import scQUEST as sp
from scQUEST._data import DS

from typing import Optional

from pathlib import Path

p = Path('//bench/benchmark.out').expanduser()
# %%
ad = sp.dataset.breastCancerAtlasRaw()

x_train, x_test = train_test_split(ad.X[ad.obs.tissue_type == 'N'], test_size=.3)
d1 = x_train.shape[1]


# %%

def get_keras_model(d1, d2=10, d3=2):
    input_data = Input(shape=(d1,))
    encoded1 = Dense(d2, activation='relu')(input_data)
    encoded2 = Dense(d3, activation='relu')(encoded1)
    decoded1 = Dense(d2, activation='relu')(encoded2)
    decoded2 = Dense(d1, activation='sigmoid')(decoded1)
    ae = Model(input_data, decoded2)
    return ae


def get_torch_model(d1, d2=10, d3=2):
    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(d1, d2), nn.ReLU(),
                                       nn.Linear(d2, d3), nn.ReLU(),
                                       nn.Linear(d3, d2), nn.ReLU(),
                                       nn.Linear(d2, d1), nn.Sigmoid())

        def forward(self, x):
            return self.model(x)

    return Model()


def get_torch_model_moduleList(d1, d2=10, d3=2):
    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.activation = nn.ReLU()
            self.model = nn.ModuleList([nn.Linear(d1, d2),
                                        nn.Linear(d2, d3),
                                        nn.Linear(d3, d2),
                                        nn.Linear(d2, d1)])

        def forward(self, x):
            for layer in self.model[:-1]:
                x = self.activation(layer(x))
            return torch.sigmoid(self.model[-1](x))

    return Model()


def get_lightning_model(d1, d2=10, d3=2):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.loss = nn.MSELoss()
            self.logging = False
            self.learning_rate = 1e-3
            self.model = get_torch_model(d1, d2, d3)

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


def get_larger_lightning_model(d1, d2=10, d3=2):
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


# %%

def fit_keras(model, x_train, x_test, epochs=10, batch_size=256):
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(x_train, x_train,
              epochs=epochs, verbose=1,
              batch_size=batch_size,
              shuffle=True,
              validation_data=None)
    # model.evaluate(x_test, x_test, verbose=1)


def fit_torch(model, x_train, x_test, epochs=10, batch_size=256):
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    x_train_ds = DS(x_train)
    x_test_ds = DS(x_test)
    train_dataloader = DataLoader(x_train_ds, batch_size=batch_size, num_workers=n_cpu)

    test_dataloader = DataLoader(x_test_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, nn.MSELoss(), optimizer)
        # test_loop(test_dataloader, model, nn.MSELoss())


def fit_lightning(model, x_train, x_test, epochs=10, batch_size=256):
    x_train_ds = DS(x_train)
    x_test_ds = DS(x_test)
    train_dataloader = DataLoader(x_train_ds, batch_size=batch_size, num_workers=n_cpu)
    test_dataloader = DataLoader(x_test_ds, batch_size=batch_size, num_workers=n_cpu)

    trainer = pl.Trainer(logger=False, max_epochs=epochs, accelerator='cpu', precision=16, enable_checkpointing=False)
    trainer.fit(model, train_dataloader=train_dataloader)
    # trainer.test(test_dataloaders=test_dataloader)


# # %% keras
# model_kears = get_keras_model(d1)
# fit_keras(model_kears, x_train, x_test)
#
# # %% torch
# model = get_torch_model(d1)
# fit_torch(model, x_train, x_test)
#
# # %% torch module
# model = get_torch_model_moduleList(d1)
# fit_torch(model, x_train, x_test)
#
# # %% lightning
# model = get_lightning_model(d1)
# fit_lightning(model, x_train, x_test)

# %%
import time

# n_cpu = os.cpu_count()
n_cpu = 0
n = 5
epochs = 3
res = dict(keras=[], torch=[], torch_module=[], lightning=[], large_keras=[], large_torch=[])

for i in range(n):
    start = time.time()
    model = get_keras_model(d1)
    fit_keras(model, x_train, None, epochs=epochs)
    res['keras'].append((time.time() - start) / epochs)

for i in range(n):
    start = time.time()
    model = get_torch_model(d1)
    fit_torch(model, x_train, None, epochs=epochs)
    res['torch'].append((time.time() - start) / epochs)

for i in range(n):
    start = time.time()
    model = get_torch_model_moduleList(d1)
    fit_torch(model, x_train, None, epochs=epochs)
    res['torch_module'].append((time.time() - start) / epochs)

for i in range(n):
    start = time.time()
    model = get_lightning_model(d1)
    fit_lightning(model, x_train, None, epochs=epochs)
    res['lightning'].append((time.time() - start) / epochs)

for i in range(n):
    start = time.time()
    model = get_large_torch_model(d1)
    fit_torch(model, x_train, None, epochs=epochs)
    res['large_torch'].append((time.time() - start) / epochs)

for i in range(n):
    start = time.time()
    model = get_large_keras_model(d1)
    fit_keras(model, x_train, None, epochs=epochs)
    res['large_keras'].append((time.time() - start) / epochs)

for i in range(n):
    start = time.time()
    model = get_lightning_model(d1)
    fit_lightning(model, x_train, None, epochs=epochs)
    res['larger_lightning'].append((time.time() - start) / epochs)

# %%
import numpy as np

with open(p, 'a') as f:
    for key, item in res.items():
        if len(item) == 0: continue
        s = f'{key}: mean: {np.mean(item):.2f}, max: {np.max(item):.2f}, min: {np.min(item):.2f}'
        # print(f'{key}: mean: {np.mean(item):.2f}, max: {np.max(item):.2f}, min: {np.min(item):.2f}')
        f.write(s)
        f.write('\n')
