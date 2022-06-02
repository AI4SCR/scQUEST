import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from anndata import AnnData
from scipy.sparse import issparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import torch

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Optional

TRAIN_DATALOADERS = EVAL_DATALOADERS = DataLoader


class DS(Dataset):
    def __init__(self, x):
        self.data = x

    def __getitem__(self, item):
        return self.data[item], self.data[item]

    def __len__(self):
        return len(self.data)


class AnnDatasetClf(Dataset):
    def __init__(
        self,
        ad: AnnData,
        target: str,
        layer: Optional[str] = None,
        encode_targets_one_hot: bool = False,
    ) -> None:
        super(AnnDatasetClf, self).__init__()
        self.ad = ad
        self.target = target
        self.layer = layer

        data = ad.X if layer is None else ad.layers[layer]
        self.data = (
            torch.tensor(data.A).float()
            if issparse(data)
            else torch.tensor(data).float()
        )

        targets = torch.tensor(ad.obs[target])
        if (not targets.min() == 0) or (
            not targets.max() == len(torch.unique(targets)) - 1
        ):
            raise ValueError(
                "targets need to be (numeric) in the range [0,C) where C is the number of classes. Please encode your target accordingly (for example with pandas `ngroup()`)"
            )
        # ATTENTION: pytorch crossentropy interprest one_hot encoded targets as class probabilities!
        self.targets = (
            F.one_hot(targets).float() if encode_targets_one_hot else targets.long()
        )

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
        self.data = (
            torch.tensor(data.A).float()
            if issparse(data)
            else torch.tensor(data).float()
        )

    def __getitem__(self, item) -> (torch.TensorType, torch.TensorType):
        return self.data[item], self.data[item]

    def __len__(self) -> int:
        return len(self.data)


class AnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ad: AnnData,
        target: str,
        ad_dataset_cls,
        layer: Optional[str] = None,
        test_size: float = 0.10,
        validation_size: float = 0.1,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ):
        super(AnnDataModule, self).__init__()

        self.ad = ad
        self.target = target
        self.layer = layer
        self.ad_dataset_cls = ad_dataset_cls
        self.test_size = test_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.seed = seed if seed else 42

        self.dataset = ad_dataset_cls(ad, target=target, layer=layer)

        # dataset indices
        if target is None:
            # autoencoder
            self.sss_train_test = ShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.seed
            )
            self.sss_fit_val = ShuffleSplit(
                n_splits=1, test_size=self.validation_size, random_state=self.seed
            )

            self.train_idx, self.test_idx = next(
                self.sss_train_test.split(np.zeros(len(self.dataset)))
            )
            self.fit_idx, self.val_idx = next(
                self.sss_fit_val.split(np.zeros(len(self.train_idx)))
            )
        else:
            # classification
            self.sss_train_test = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.seed
            )
            self.sss_fit_val = StratifiedShuffleSplit(
                n_splits=1, test_size=self.validation_size, random_state=self.seed
            )

            self.train_idx, self.test_idx = next(
                self.sss_train_test.split(np.zeros(len(self.dataset)), ad.obs[target])
            )
            self.fit_idx, self.val_idx = next(
                self.sss_fit_val.split(
                    np.zeros(len(self.train_idx)), ad.obs[target][self.train_idx]
                )
            )

        self.fit_idx = self.train_idx[self.fit_idx]
        self.val_idx = self.train_idx[self.val_idx]

        # datasets
        self.train = None
        self.fit = None
        self.val = None
        self.test = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = Subset(self.dataset, self.train_idx)

        if stage in ("fit", None):
            # here we subset the generated training indices with the generated fitting and validation indices
            self.fit, self.val = Subset(self.dataset, self.fit_idx), Subset(
                self.dataset, self.val_idx
            )

        if stage in ("test", None):
            self.test = Subset(self.dataset, self.test_idx)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.fit, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
