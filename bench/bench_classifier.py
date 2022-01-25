# %%
from pathlib import Path

import starProtocols as sp
from starProtocols.tests import dummy_annData

# %%
f_ad_labeled = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg_subset_clusterlabeled/ad_labelled.h5ad')
f_ad_unlabeled = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg_subset_clusterlabeled/ad_unlabelled.h5ad')

# ad_train = anndata.read_h5ad(f_ad_labeled)
# ad_pred = anndata.read_h5ad(f_ad_unlabeled)
ad_train = ad_pred = dummy_annData()

ad_train.obs['y_id'] = ad_train.obs.groupby('y').ngroup()

# %%

clf = sp.EpithelialClassifier()
clf.fit(ad_train, 'y_id')
clf.predict(ad_pred)

# %% with pre-processing
preprocessing = [sp.StandardScale()]
clf = sp.EpithelialClassifier()
clf.fit(ad_train, 'y_id', preprocessing=preprocessing)
clf.predict(ad_pred)

# %%

# NOTE: Discuss default classifier architecture
# - Early stopping: 10 epochs unchanged, was at epoch 254
# -

# %%

clf = sp.EpithelialClassifier(n_in=ad_train.shape[1])
clf.fit(ad_train, target='y_id')
clf.predict(ad_pred)

# %%
from starProtocols import DefaultClassifier

clf = DefaultClassifier(10)

# %%
from starProtocols import LitModule

module = LitModule(clf)

# %%
from starProtocols import AnnDataset, dummy_annData

ad = dummy_annData()
ds = AnnDataset(ad, 'y')
ds = AnnDataset(ad, 'y', encode_targets_one_hot=False)

# %%
from starProtocols import StandardScale, dummy_annData, AnnDataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import pandas as pd

ad = dummy_annData()
ds = AnnDataset(ad, 'y')
pd.DataFrame(ds.data.numpy()).hist(bins=100);
plt.show()

ss = StandardScale()
ss.fit(ds)

ds_norm = ss.fit_transform(ds)
ss.fit_transform(ds, inplace=True)

ad = dummy_annData()
ds = AnnDataset(ad, 'y')
a, b = random_split(ds, [1000] * 2)
ss = StandardScale()
ss.fit_transform(a)
ss.transform(b)

# %%
from starProtocols import AnnDataModule, dummy_annData

ad = dummy_annData()
dm = AnnDataModule(ad, 'y')
dm.setup()

# %%
from starProtocols import AnnDataModule, dummy_annData
from starProtocols import StandardScale

ad = dummy_annData()
dm = AnnDataModule(ad, 'y')

preprocessing = [StandardScale()]
dm_norm = AnnDataModule(ad, 'y', preprocessing=preprocessing)
dm_norm.setup()

# %%
import pytorch_lightning as pl
from starProtocols import DefaultClassifier, LitModule, dummy_annData, AnnDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

ad = dummy_annData()
model = DefaultClassifier(n_in=ad.shape[1])
clf = LitModule(model)
dm = AnnDataModule(ad, 'y', preprocessing=None)

trainer = pl.Trainer(logger=False, callbacks=None, max_epochs=3)
trainer.fit(model=clf, datamodule=dm)

# %%
callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=False, min_delta=1e-3, patience=0,
                           check_on_train_epoch_end=False)]
trainer = pl.Trainer(logger=False, callbacks=callbacks, max_epochs=30)
trainer.fit(model=clf, datamodule=dm)
trainer.test(model=clf, datamodule=dm)
# %%
from starProtocols import dummy_annData, EpithelialClassifier

ad_train = ad_pred = dummy_annData()
ad_train.obs['y_id'] = ad_train.obs.groupby('y').ngroup()
clf = EpithelialClassifier()
clf.fit(ad_train, 'y_id')
clf.predict(ad_pred)

# %% low-level call
from starProtocols import DefaultClassifier, LitModule, dummy_annData, AnnDataModule, EpithelialClassifier, \
    StandardScale
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

ad_train = ad_pred = dummy_annData()
ad_train.obs['y_id'] = ad_train.obs.groupby('y').ngroup()

dm = AnnDataModule(ad_train, 'y_id')

model = DefaultClassifier(n_in=ad_train.shape[1])
module = LitModule(model=model)

clf = EpithelialClassifier(model=module)

preprocessing = [StandardScale()]
clf.fit(datamodule=dm,
        early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=1e-3))
