# %%
import starProtocols as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve
from functools import partial

# %% load annotated data
ad_anno = sp.dataset.breastCancerAtlas()
marker = sp.utils.DEFAULT_MARKER_CLF

# %% plotting config
marker_plot_fnc = partial(sns.histplot, bins=50)
# marker_plot_fnc = partial(sns.kdeplot, bw_method=1)
alpha = .2

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

# %% marker distribution
N = 100000
cmap = plt.get_cmap('tab10')
c1, c2 = cmap(1), cmap(2)

epi = ad_anno[ad_anno.obs.is_epithelial == 1].layers[layer_name]
notEpi = ad_anno[ad_anno.obs.is_epithelial == 0].layers[layer_name]

idx = np.random.randint(0, len(epi), size=N)
epi = epi[idx,]
idx = np.random.randint(0, len(notEpi), size=N)
notEpi = notEpi[idx,]

nrow, ncol = 6, 6
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
for _epi, _notEpi, ax, title in zip(epi.T, notEpi.T, axs.flat, ad_anno.var.desc):
    line1 = marker_plot_fnc(x=_epi, ax=ax, color=c1)
    line1 = marker_plot_fnc(x=_notEpi, ax=ax, color=c2, alpha=alpha)
    ax.set_title(title)

l1, = ax.plot([1, 2, 3], label='epithelial', c=c1)
l2, = ax.plot([1, 2, 3], label='non-epithelial', c=c2)
fig.legend(handles=[l1, l2])
fig.suptitle('Marker Distribution of (Non-)Epithelial Cells')
fig.tight_layout()
fig.show()

# %% train classifier
clf = sp.EpithelialClassifier(n_in=ad_anno.shape[1], seed=1)
clf.fit(ad_anno, layer=layer_name, target='is_epithelial', max_epochs=1, seed=1)

# %% loss classifier
hist = clf.logger.history
fit_loss = pd.DataFrame.from_records(hist['fit_loss'], columns=['step', 'loss']).assign(stage='fit')
val_loss = pd.DataFrame.from_records(hist['val_loss'], columns=['step', 'loss']).assign(stage='validation')
test_loss = pd.DataFrame.from_records(hist['test_loss'], columns=['step', 'loss']).assign(stage='test')
loss = pd.concat((fit_loss, val_loss, test_loss))
loss = loss.reset_index(drop=True)

fig, ax = plt.subplots()
for stage in ['fit', 'validation', 'test']:
    if stage == 'test':
        ax.plot([0, ax.get_xlim()[1]], [loss[loss.stage == stage].loss, loss[loss.stage == stage].loss], label='test')
    else:
        ax.plot(loss[loss.stage == stage].step, loss[loss.stage == stage].loss, 'o-', label=stage)
fig.legend()
fig.show()

# %% confusion matrix
dl = clf.datamodule.test_dataloader()
data = dl.dataset.dataset.data
y = dl.dataset.dataset.targets.argmax(axis=1)
yhat = clf.model(data)

m = confusion_matrix(y, yhat, normalize='pred')
m = pd.DataFrame(m, columns=['non-epithelial', 'epithelial'], index=['non-epithelial', 'epithelial'])
m.index.name = 'true'
m.columns.name = 'predicted'
sns.heatmap(m.T * 100, annot=True, fmt='.3f');
plt.show()

# %% plot ROC curve
yhat = clf.model(data).numpy()
yhat1 = clf.model.model(data).detach().numpy()  # here we access the actual pytorch model
tmp = np.array([yhat1[i][yhat[i]] for i in range(len(yhat))])
score = np.zeros_like(yhat, dtype=float) - 1
score[yhat == 1] = tmp[yhat == 1]
score[yhat == 0] = (1 - tmp[yhat == 0])
fpr, tpr, thresholds = roc_curve(y, score)
plt.plot(fpr, tpr, '-o')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot([0, 1], [0, 1], 'b:')
plt.title('ROC Curve')
plt.show()

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

# %% plot marker distribution
N = 100000
cmap = plt.get_cmap('tab10')
c1, c2 = cmap(1), cmap(2)

for cell_type in ['epithelial', 'non-epithelial']:
    is_epithelial = 1 if cell_type == 'epithelial' else 0
    y = ad_anno[ad_anno.obs.is_epithelial == is_epithelial].layers[layer_name]
    yhat = ad_pred[ad_pred.obs.clf_is_epithelial == is_epithelial].layers[layer_name]

    idx = np.random.randint(0, len(y), size=N)
    y = y[idx,]
    idx = np.random.randint(0, len(yhat), size=N)
    yhat = yhat[idx,]

    nrow, ncol = 6, 6
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    for _y, _yhat, ax, title in zip(y.T, yhat.T, axs.flat, ad_anno.var.desc):
        line1 = marker_plot_fnc(x=_y, ax=ax, color=c1)
        line2 = marker_plot_fnc(x=_yhat, ax=ax, alpha=alpha, color=c2)
        ax.set_title(title)

    l1, = ax.plot([1, 2, 3], label='anno', c=c1)
    l2, = ax.plot([1, 2, 3], label='pred', c=c2)
    fig.suptitle(cell_type)
    fig.legend(handles=[l1, l2])
    fig.tight_layout()
    fig.show()

# %%
del ad_pred
del ad_anno
# %% plot fraction of epithelial cells
data = ad.obs.copy()
frac_epith = data.groupby(
    ['tissue_type', 'breast', 'patient_number']).is_epithelial.mean().dropna().reset_index()
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
markers = set(
    ['Vol7', 'H3K27me3', 'K5', 'PTEN', 'CD44', 'K8K18', 'cMYC', 'SMA', 'CD24', 'HER2', 'AR', 'BCL2', 'p53',
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
ad_train = ad[
    (ad.obs.patient_number.isin(patients)) & (ad.obs.tissue_type == 'N') & (ad.obs.is_epithelial == 1),
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
Abn.fit(ad_train, layer=layer_name, max_epochs=100)

del ad_train
# %% loss classifier
hist = Abn.logger.history
fit_loss = pd.DataFrame.from_records(hist['fit_loss'], columns=['step', 'loss']).assign(stage='fit')
val_loss = pd.DataFrame.from_records(hist['val_loss'], columns=['step', 'loss']).assign(stage='validation')
test_loss = pd.DataFrame.from_records(hist['test_loss'], columns=['step', 'loss']).assign(stage='test')
loss = pd.concat((fit_loss, val_loss, test_loss))
loss = loss.reset_index(drop=True)

fig, ax = plt.subplots()
for stage in ['fit', 'validation', 'test']:
    if stage == 'test':
        ax.plot([0, ax.get_xlim()[1]], [loss[loss.stage == stage].loss, loss[loss.stage == stage].loss], label='test')
    else:
        ax.plot(loss[loss.stage == stage].step, loss[loss.stage == stage].loss, 'o-', label=stage)
fig.legend()
fig.show()

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

# %% plot reconstruction error
import torch

y = ad_pred.layers[layer_name]
yhat = Abn.model.model(torch.tensor(y)).detach().numpy()  # access base torch model
err = ad_pred.layers['abnormality']

fig, axs = plt.subplots(1, 3)
for ax, dat, title in zip(axs.flat, [y, yhat, err], ['Input', 'Reconstruction', 'Error']):
    ax: plt.Axes
    cmap = 'seismic' if title == 'Error' else 'viridis'
    im = ax.imshow(dat, cmap=cmap)
    ax.set_aspect(dat.shape[1] * 3 / dat.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('features')
    ax.set_title(title)
fig.colorbar(im)
axs[0].set_ylabel('cells')
fig.show()

# %% plot abnormality
dat = ad_pred.obs.groupby(['tissue_type', 'breast', 'patient_number']).abnormality.agg(
    np.median).dropna().reset_index()

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

mapping = ad_indiv.obs[['tissue_type', 'breast', 'patient_number', 'sample_id']].set_index(
    'sample_id').to_dict()
dat['breast'] = dat.index.map(mapping['breast'])
dat['tissue_type'] = dat.index.map(mapping['tissue_type'])
dat['patient_number'] = dat.index.map(mapping['patient_number'])

# %%

fig, ax = plt.subplots()
sns.boxplot(data=dat, x='tissue_type', y='individuality', ax=ax, whis=[0, 100])
sns.stripplot(data=dat, x='tissue_type', y='individuality', ax=ax)
fig.show()
