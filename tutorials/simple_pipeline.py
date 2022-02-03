# %%
import numpy as np

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

# %%
clf = sp.EpithelialClassifier()
clf.fit(ad, target='is_epithelial', max_epochs=300)
clf.predict(ad)

# %%
abnormality = sp.Abnormality()
# abnormality.fit(ad)
abnormality.predict(ad)
abnormality.aggregate(ad)

# %%
indiv = sp.Individuality()
indiv.predict(ad, labels=ad.obs.celltype)
ad.uns['individuality_agg']
ad.obsm['individuality']
