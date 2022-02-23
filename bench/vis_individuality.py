# %%
import scQUEST as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
ad = sp.dataset.breastCancerAtlasRaw()
marker = sp._utils.DEFAULT_MARKER_CLF

# %%
mask = []
for m in marker:
    mask.append(ad.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad.var['used_in_clf'] = mask
ad = ad[:, ad.var.used_in_clf]

# %% pre-processing

X = ad.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'arcsinh'
ad.layers[layer_name] = X

# %%
tmp = ad.obs.groupby(['tissue_type', 'breast', 'patient_number']).indices
indices = []
for key, item in tmp.items():
    size = min(len(item), 100)
    idx = np.random.randint(0, len(item), size)
    indices.extend(item[idx])
indices = np.array(indices)
ad = ad[indices]

# %% sample indicator
ad.obs['sample'] = ad.obs.groupby(['tissue_type', 'breast', 'patient_number']).ngroup()

# %%
indiv = sp.Individuality()
indiv.predict(ad, ad.obs['sample'], layer=layer_name)

# %%
dat = ad.uns['individuality_agg'].copy()
dat = pd.DataFrame(np.diag(dat), index=dat.index, columns=['individuality'])
# ad.obsm['individuality']

mapping = ad.obs[['tissue_type', 'breast', 'patient_number', 'sample']].set_index('sample').to_dict()
dat['breast'] = dat.index.map(mapping['breast'])
dat['tissue_type'] = dat.index.map(mapping['tissue_type'])
dat['patient_number'] = dat.index.map(mapping['patient_number'])

# %%
# dat = dat.groupby(['tissue_type', 'breast', 'patient_number']).individuality.agg(np.median).dropna().reset_index()

fig, ax = plt.subplots()
sns.boxplot(data=dat, x='tissue_type', y='individuality', ax=ax, whis=[0, 100])
sns.stripplot(data=dat, x='tissue_type', y='individuality', ax=ax)
fig.show()
