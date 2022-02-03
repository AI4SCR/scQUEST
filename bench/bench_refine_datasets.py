# %%
import anndata
from pathlib import Path

# %%
f_raw = Path('/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg/ad.h5ad')
f_lab = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg_subset_clusterlabeled/ad_labelled.h5ad')
ad = anndata.read_h5ad(f_lab)
ad.obs['cluster'] = ad.obs.cluster.astype(int).astype('category')
