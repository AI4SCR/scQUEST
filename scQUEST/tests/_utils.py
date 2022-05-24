# %%
import torch
import pandas as pd
from anndata import AnnData


# %%
def dummy_annData(n1: int = 1000, n2: int = 1000, n_feat: int = 25):
    cls1 = torch.randn((1, n_feat)).repeat((n1, 1)) + 5
    cls2 = torch.randn((1, n_feat)).repeat((n2, 1))
    X = torch.vstack((cls1, cls2))

    y = torch.hstack((torch.tensor(0).repeat(n1), torch.tensor(1).repeat(n2)))
    obs = pd.DataFrame(
        y.numpy(), columns=["y"], index=pd.Index(range(len(y)), dtype="str")
    )
    obs["y_id"] = y.numpy()
    return AnnData(X=X.numpy(), obs=obs)
