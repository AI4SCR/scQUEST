# %%
import numpy as np
import matplotlib.pyplot as plt

import starProtocols as sp
import anndata
from pathlib import Path

# %% load data
# f_lab = Path(
#     '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg_subset_clusterlabeled/ad_labelled.h5ad')
# ad = anndata.read_h5ad(f_lab)

ad = sp.dataset.breastCancerAtlas()
mask = np.zeros(ad.shape[1]).astype(bool)
for i in sp.utils.DEFAULT_MARKERS:
    tmp = ad.var.desc.str.contains('_' + i)
    mask = mask | tmp

ad = ad[:, mask]
ad.obs['cluster'] = ad.obs.cluster.astype(int).astype('category')
ad.obs['is_epithelial'] = (ad.obs.celltype_class == 'epithelial').astype(int)

# %% pre-processing

# censoring
q001 = np.quantile(ad.X, .001, axis=0)
q999 = np.quantile(ad.X, .999, axis=0)

X = ad.X.copy()
for i, qmin, qmax in zip(range(X.shape[1]), q001, q999):
    # X[X[:, i] < qmin, i] = qmin
    X[X[:, i] > qmax, i] = qmax

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

lname = 'censor_arcsinh_norm'
ad.layers[lname] = X

# %% visualise distributions

for X in [ad.X.T, ad.layers[lname].T]:
    nrow, ncol = 9, 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 2))
    for ax, dat, title in zip(axs.flat, X, ad.var.desc):
        g = ax.hist(dat, bins=50)
        ax.set_title(title)
    fig.tight_layout()
    fig.show()

# %%
fig, ax = plt.subplots()
dat = ad.layers[lname][:, ad.var.desc.str.contains('K8K18')]
dat = dat[ad.obs.celltype_class == 'epithelial']
g = ax.hist(dat.ravel(), bins=50)
fig.show()

# %%
ad = ad[:1000, ]

# %%

clf = sp.EpithelialClassifier(seed=1)
clf.fit(ad, layer=lname, target='is_epithelial', max_epochs=5, seed=1)
clf.predict(ad)

# %%
abnormality = sp.Abnormality()
abnormality.fit(ad)
abnormality.predict(ad)
abnormality.aggregate(ad)

# %%
indiv = sp.Individuality()
indiv.predict(ad, labels=ad.obs.celltype)
ad.uns['individuality_agg']
ad.obsm['individuality']
