# %%
import scQUEST as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

f_lab = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg/ad.h5ad')

# %% load annotated data
ad_anno = sp.dataset.breastCancerAtlas()
marker = sp.utils.DEFAULT_MARKER_CLF

# %% define the markers used in the classifier
mask = []
for m in marker:
    mask.append(ad_anno.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad_anno.var['used_in_clf'] = mask

# %% pre-processing
ad_anno = ad_anno[:, ad_anno.var.used_in_clf]
ad_anno.obs['is_epithelial'] = (ad_anno.obs.celltype_class == 'epithelial').astype(int)

X = ad_anno.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'arcsinh'
ad_anno.layers[layer_name] = X

# %% train classifier
clf = sp.EpithelialClassifier(n_in=ad_anno.shape[1], seed=1)
clf.fit(ad_anno, layer=layer_name, target='is_epithelial', max_epochs=10, seed=1)
del ad_anno
# %% prepare the whole dataset for celltype classification

ad = sp.dataset.breastCancerAtlasRaw()

mask = []
for m in marker:
    mask.append(ad.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad.var['used_in_clf'] = mask

ad_pred = ad[:, ad.var.used_in_clf]

X = ad_pred.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'arcsinh'
ad_pred.layers[layer_name] = X

# %%
clf.predict(ad_pred, layer=layer_name)
ad.obs['is_epithelial'] = ad_pred.obs.clf_is_epithelial.values
del ad_pred

# %% plot fraction of epithelial cells
data = ad.obs.copy()
frac_epith = data.groupby(['tissue_type', 'breast', 'patient_number']).is_epithelial.mean().dropna().reset_index()
fig, ax = plt.subplots()
sns.boxplot(y='is_epithelial', x='tissue_type', data=frac_epith, ax=ax, whis=[0, 100])
sns.stripplot(y='is_epithelial', x='tissue_type', data=frac_epith, ax=ax, color=".3")
fig.show()

# %% prepare dataset for abnormality

patients = ['N_BB013', 'N_BB028', 'N_BB034', 'N_BB035', 'N_BB037', 'N_BB046', 'N_BB051', 'N_BB055', 'N_BB058',
            'N_BB064',
            'N_BB065', 'N_BB072', 'N_BB073', 'N_BB075', 'N_BB076', 'N_BB090', 'N_BB091', 'N_BB093', 'N_BB094',
            'N_BB096',
            'N_BB099', 'N_BB101', 'N_BB102', 'N_BB110', 'N_BB120', 'N_BB131', 'N_BB144', 'N_BB147', 'N_BB153',
            'N_BB154',
            'N_BB155', 'N_BB167', 'N_BB192', 'N_BB194', 'N_BB197', 'N_BB201', 'N_BB204', 'N_BB209', 'N_BB210',
            'N_BB214',
            'N_BB221']
patients = [i.split('N_')[1] for i in patients]
markers = set(['Vol7', 'H3K27me3', 'K5', 'PTEN', 'CD44', 'K8K18', 'cMYC', 'SMA', 'CD24', 'HER2', 'AR', 'BCL2', 'p53',
               'EpCAM', 'CyclinB1',
               'PRB', 'CD49f', 'Survivin', 'EZH2', 'Vimentin', 'cMET', 'AKT', 'ERa', 'CA9', 'ECadherin', 'Ki67', 'EGFR',
               'K14', 'HLADR', 'K7', 'panK'])
drop_markers = set(['Vol7', 'H3K27me3', 'CyclinB1', 'Ki67'])
markers = markers - drop_markers

# %%
mask = []
for m in markers:
    mask.append(ad.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad.var['used_in_abnormality'] = mask

# %% subset whole dataset to only include epithelial cells from selected patients and normal tissue
ad_train = ad[(ad.obs.patient_number.isin(patients)) & (ad.obs.tissue_type == 'N') & (ad.obs.is_epithelial == 1),
              ad.var.used_in_abnormality]

X = ad_train.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

# min-max normalisation
minMax = MinMaxScaler()
X = minMax.fit_transform(X)

layer_name = 'arcsinh_norm'
ad_train.layers[layer_name] = X

# %% train AE for abnormality
Abn = sp.Abnormality(n_in=ad_train.shape[1])
Abn.fit(ad_train, layer=layer_name, max_epochs=5)
del ad_train

# %% pre-process for Abnormality prediction
ad_pred = ad[ad.obs.is_epithelial == 1, ad.var.used_in_abnormality]

X = ad_pred.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

# min-max normalisation
X = minMax.transform(X)

layer_name = 'arcsinh_norm'
ad_pred.layers[layer_name] = X

# %% estimate reconstruction error
Abn.predict(ad_pred, layer=layer_name)  # where should I save the results in the AnnData? layers or obsm?
mse = (ad_pred.layers['abnormality'] ** 2).mean(axis=1)

ad_pred.obs['abnormality'] = mse

# %% plot abnormality
dat = ad_pred.obs.groupby(['tissue_type', 'breast', 'patient_number']).abnormality.agg(np.median).dropna().reset_index()

fig, ax = plt.subplots()
sns.boxplot(data=dat, x='tissue_type', y='abnormality', ax=ax, whis=[0, 100])
sns.stripplot(data=dat, x='tissue_type', y='abnormality', ax=ax)
fig.show()

# %% individuality preprocessing
ad_indiv = ad[:, ad.var.used_in_abnormality]
X = ad_indiv.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'arcsinh'
ad_indiv.layers[layer_name] = X

ad_indiv.obs['sample_id'] = ad_indiv.obs.groupby(['tissue_type', 'breast', 'patient_number']).ngroup()

# %% sub-sample
tmp = ad_indiv.obs.groupby(['sample_id']).indices
n_cells = 100
indices = []
for key, item in tmp.items():
    size = min(len(item), n_cells)
    idx = np.random.randint(0, len(item), size)
    indices.extend(item[idx])
indices = np.array(indices)
ad_indiv = ad_indiv[indices]

# %%
Indiv = sp.Individuality()
Indiv.predict(ad_indiv, ad_indiv.obs.sample_id, layer=layer_name)

# ad.obsm['individuality']
# ad.uns['individuality_agg']

# %%
dat = ad_indiv.uns['individuality_agg'].copy()
dat = pd.DataFrame(np.diag(dat), index=dat.index, columns=['individuality'])
# ad.obsm['individuality']

mapping = ad_indiv.obs[['tissue_type', 'breast', 'patient_number', 'sample_id']].set_index('sample_id').to_dict()
dat['breast'] = dat.index.map(mapping['breast'])
dat['tissue_type'] = dat.index.map(mapping['tissue_type'])
dat['patient_number'] = dat.index.map(mapping['patient_number'])

# %%

fig, ax = plt.subplots()
sns.boxplot(data=dat, x='tissue_type', y='individuality', ax=ax, whis=[0, 100])
sns.stripplot(data=dat, x='tissue_type', y='individuality', ax=ax)
fig.show()
