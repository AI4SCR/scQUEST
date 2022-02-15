# %%
import starProtocols as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from pathlib import Path

f_lab = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg/predictions.csv')

# %%
ad = sp.dataset.breastCancerAtlasRaw()
pred = pd.read_csv(f_lab)

ad.obs['clf_is_epithelial'] = pred.clf_is_epithelial.values
# %%
# grpC = ad.obs.groupby(['tissue_type', 'breast', 'patient_number']).size()
# ad[ad.obs.tissue_type == 'H'].obs.groupby(['tissue_type', 'breast', 'patient_number']).size()

# %% plot fraction of epithelial cells
data = ad.obs.copy()
frac_epith = data.groupby(['tissue_type', 'breast', 'patient_number']).clf_is_epithelial.mean().dropna().reset_index()
fig, ax = plt.subplots()
sns.boxplot(y='clf_is_epithelial', x='tissue_type', data=frac_epith, ax=ax, whis=[0, 100])
sns.stripplot(y='clf_is_epithelial', x='tissue_type', data=frac_epith, ax=ax, color=".3")
fig.show()

# %% plot training progress

fig, axs = plt.subplots(1, 2)
log = clf.logger.history

d = log['fit_loss']
axs[0].plot(range(len(d)), d, label='fit_loss')

n = len(log['train_metric_accuracy'])
axs[1].plot(range(n), log['train_metric_accuracy'], label='train_metric_accuracy')
d = log['test_metric_accuracy']
axs[1].plot(range(n), np.repeat(log['test_metric_accuracy'], n), label='test_metric_accuracy')
axs[0].legend()
axs[1].legend()
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

# %% plot cross-validation

acc = []
prec = []
seeds = [49, 42, 11111, 39, 4309578]
for i in seeds:
    clf = sp.EpithelialClassifier(seed=i)
    clf.fit(ad_train, layer=layer_name, target='is_epithelial', max_epochs=3, seed=i)
    acc.append(clf.logger.history['test_metric_accuracy'])
    prec.append(clf.logger.history['test_metric_precision'])

acc = [i[0] for i in acc]
fig, ax = plt.subplots()
ax.bar(height=acc, x=range(len(seeds)))
ax.set_title('Cross-Validation')
ax.set_xlabel('Seed')
ax.set_ylabel('Accuracy')
fig.show()

# %% plot distributions
N = 50000
for i, suptitle in zip([0, 1], ['non-epithelial', 'epithelial']):
    d1 = ad_train[ad_train.obs.is_epithelial == i].layers[layer_name]
    d2 = ad[ad.obs.clf_is_epithelial == i].layers[layer_name]

    d1 = d1[np.random.randint(0, len(d1), N)]
    d2 = d2[np.random.randint(0, len(d2), N)]

    X = np.vstack((d1, d2)).T
    hue = np.repeat(['manual', 'ann'], repeats=[len(d1), len(d2)])

    nrow, ncol = 6, 6
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 2))
    for ax, dat, title in zip(axs.flat, X, ad_train.var.desc):
        # g = sns.kdeplot(x=dat, hue=hue, ax=ax)
        g = sns.histplot(x=dat, hue=hue, ax=ax, bins=50, stat='probability')
        ax.set_title(title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.show()
