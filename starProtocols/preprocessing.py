from torch.utils.data import Subset
from torch.utils.data import Dataset


class Preprocessor:

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, dataset: Dataset):
        pass

    def transform(self, dataset: Dataset, inplace: bool):
        pass

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)


class StandardScale(Preprocessor):

    def __init__(self, with_mean: bool = True, with_var: bool = True):
        super(StandardScale, self).__init__()
        self.with_mean = with_mean
        self.with_var = with_var
        self.mean_ = None
        self.std_ = None

    def fit(self, dataset: Dataset):
        data = self.get_data(dataset)
        self.mean_ = data.mean(0, keepdim=True)
        self.std_ = data.std(0, unbiased=False, keepdim=True)

    def transform(self, dataset: Dataset):
        data = self.get_data(dataset)
        data = (data - self.mean_) / self.std_
        return (data - self.mean_) / self.std_

    def get_data(self, dataset):
        if isinstance(dataset, Dataset):
            return dataset.data
        elif isinstance(dataset, Subset):
            return dataset.dataset.data


class censore(Preprocessor):
    pass
