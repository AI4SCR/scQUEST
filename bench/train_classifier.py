# %%
import scQUEST as sp
import pandas as pd
import numpy as np

from pathlib import Path

f_lab = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg/predictions.csv')

# %%
ad_train = sp.dataset.breastCancerAtlas()

# %%
marker = sp.utils.DEFAULT_MARKER_CLF

mask = []
for m in marker:
    mask.append(ad_train.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad_train.var['used_in_clf'] = mask

# %% pre-processing
ad_train = ad_train[:, ad_train.var.used_in_clf]
ad_train.obs['is_epithelial'] = (ad_train.obs.celltype_class == 'epithelial').astype(int)

X = ad_train.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'arcsinh'
ad_train.layers[layer_name] = X

# %% train classifier
clf = sp.EpithelialClassifier(seed=1)
clf.fit(ad_train, layer=layer_name, target='is_epithelial', max_epochs=20, seed=1)
del ad_train
# %% predict phenotypes - preprocessing

ad = sp.dataset.breastCancerAtlasRaw()
marker = sp.utils.DEFAULT_MARKER_CLF

mask = []
for m in marker:
    mask.append(ad.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad.var['used_in_clf'] = mask

ad = ad[:, ad.var.used_in_clf]
X = ad.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'arcsinh'
ad.layers[layer_name] = X

# %%
clf.predict(ad, layer=layer_name)

# %%
ad.obs[f'clf_{clf.target}'].to_csv(f_lab)
