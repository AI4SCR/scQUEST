import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from anndata import AnnData
from scipy.sparse import issparse

from typing import Optional


class AnnDataset(Dataset):

    def __init__(self, ad: AnnData, target: Optional[str] = None, layer=None,
                 encode_targets_one_hot: bool = True) -> None:
        super(AnnDataset, self).__init__()
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
