# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Optional

import sys
import time
from pathlib import Path

p = Path('~/benchmark.out').expanduser()

# %%
mode = sys.argv[1]
data_size = int(sys.argv[2])
model_size = sys.argv[3]
epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])
N = int(sys.argv[6])
print(f'{data_size}\t{mode}\t{model_size}\t{epochs}\t{batch_size}\t{N}')
# %% data
X = torch.randn(data_size, 50)


class DataSet(Dataset):
    def __init__(self, X):
        super().__init__()
        self.data = X

    def __getitem__(self, item):
        return self.data[item], self.data[item]

    def __len__(self):
        return len(self.data)


# %% models

def small_model(d1):
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


def large_model(d1):
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


def lightning_module(model):
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.loss = nn.MSELoss()
            self.doLog = False
            self.learning_rate = 1e-3
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.doLog:
                self.log('test_loss', loss.detach())
                self.log_metrics('train', y, yhat)
            return loss

        def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
            x, y = batch
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            if self.doLog:
                self.log('test_loss', loss.detach())
                self.log_metrics('validate', y, yhat)
            return loss

        def configure_optimizers(self):
            # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

        def log_metrics(self, step, y, yhat):
            # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html

            for metric in self.metric_attrs:
                m = getattr(self, metric)
                m(yhat, y)
                self.log(f'{step}_{metric}', m)

    return Model()


# %%

def fit_torch(model, train_dataloader, epochs=10):
    def train_loop(dataloader, model, loss_fn, optimizer):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        train_loop(train_dataloader, model, nn.MSELoss(), optimizer)


def fit_lightning(model, train_dataloader, epochs=10):
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_dataloader)


# %%

ds = DataSet(X)

d1 = X.shape[-1]
model = small_model(d1) if model_size == 'small' else large_model(d1)
model = lightning_module(model) if mode == 'light' else model

res = []
for i in range(N):
    fit = fit_torch if mode == 'torch' else fit_lightning
    train_dataloader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    start = time.time()
    fit(model, train_dataloader, epochs=epochs)
    res.append((time.time() - start))

res = torch.tensor(res).float()
# %%

with open(p, 'a') as f:
    s = f'{data_size}\t{mode}\t{model_size}\t{epochs}\t{batch_size}\t{N}\t{res.mean():.2f}\t{res.max():.2f}\t{res.min():.2f}\n'
    f.write(s)
