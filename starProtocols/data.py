import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch

from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

from typing import Optional, Iterable
from .preprocessing import Preprocessor

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader

from scipy.sparse import issparse
from anndata import AnnData


class AnnDatasetClf(Dataset):

    def __init__(self, ad: AnnData, target: str, layer: Optional[str] = None,
                 encode_targets_one_hot: bool = True) -> None:
        super(AnnDatasetClf, self).__init__()
        self.ad = ad
        self.target = target
        self.layer = layer

        data = ad.X if layer is None else ad.layers[layer]
        self.data = torch.tensor(data.A).float() if issparse(data) else torch.tensor(data).float()

        targets = torch.tensor(ad.obs[target])
        self.targets = F.one_hot(targets).float() if encode_targets_one_hot else targets.float()

    def __getitem__(self, item) -> (torch.TensorType, torch.TensorType):
        return self.data[item], self.targets[item]

    def __len__(self) -> int:
        return len(self.data)


class AnnDatasetAE(Dataset):

    def __init__(self, ad: AnnData, layer=None, *args, **kwargs) -> None:
        super(AnnDatasetAE, self).__init__()
        self.ad = ad
        self.layer = layer

        data = ad.X if layer is None else ad.layers[layer]
        self.data = torch.tensor(data.A).float() if issparse(data) else torch.tensor(data).float()

    def __getitem__(self, item) -> (torch.TensorType, torch.TensorType):
        return self.data[item], self.data[item]

    def __len__(self) -> int:
        return len(self.data)


class AnnDataModule(pl.LightningDataModule):

    def __init__(self, ad: AnnData, target: str,
                 ad_dataset_cls,
                 layer: Optional[str] = None,
                 preprocessing: Optional[Iterable[Preprocessor]] = None,
                 test_size: float = 0.25, validation_size: float = 1 / 3,
                 batch_size: int = 32, seed: int = 42):
        super(AnnDataModule, self).__init__()

        self.ad = ad
        self.target = target
        self.layer = layer
        self.preprocessing = preprocessing
        self.ad_dataset_cls = ad_dataset_cls
        self.test_size = test_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.seed = seed if seed else 42

        self.dataset = ad_dataset_cls(ad, target=target, layer=layer)

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
        self._fit_preprocessors()  # TODO: Should we fit the preprocessors on the whole train or on fit,val individually?

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

    def _fit_preprocessors(self):
        if self.preprocessing:
            for pp in self.preprocessing:
                pp.fit(self.train)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
