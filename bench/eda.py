# %%
import starProtocols as sp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %%
mariannas_markers = {'Vol7', 'H3K27me3', 'K5', 'PTEN', 'CD44', 'K8K18', 'cMYC', 'SMA', 'CD24', 'HER2', 'AR', 'BCL2',
                     'p53',
                     'EpCAM', 'CyclinB1', 'PRB', 'CD49f', 'Survivin', 'EZH2', 'Vimentin', 'cMET', 'AKT', 'ERa', 'CA9',
                     'ECadherin', 'Ki67', 'EGFR', 'K14', 'HLADR', 'K7', 'panK'}

drop = {'Vol7', 'H3K27me3', 'CyclinB1', 'Ki67'}
markers = mariannas_markers - drop

# %%
ad_full = sp.dataset.breastCancerAtlas()
# excl = ['Cellvolume', 'Barcod', 'Background', 'Beads']
mask = np.zeros(ad_full.shape[1]).astype(bool)
for i in markers:
    tmp = ad_full.var.desc.str.contains('_' + i)
    mask = mask | tmp
ad = ad_full[:, mask]

# %%
df = pd.DataFrame(ad.X, index=ad.obs.index, columns=ad.var.desc)
df_desc = df.describe()
del df

# %%
m, n = 6, 5
fig, axs = plt.subplots(m, n, figsize=(3 * n, 1.5 * m))
idx = range(ad.shape[1])

bins = 100
for i, ax in zip(idx, axs.flat):
    dat = ad.X[:, i].ravel()
    q = np.quantile(dat, .99)
    dat[dat > q] = q
    title = ad.var.desc[i]
    g = ax.hist(dat, bins=bins)
    ax.set_title(title)

fig.tight_layout()
fig.show()

# %%
