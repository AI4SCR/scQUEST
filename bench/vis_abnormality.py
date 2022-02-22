# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import scQUEST as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

f_lab = Path(
    '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/Cells_CisPtneg_GDneg/predictions.csv')

# %%
ad = sp.dataset.breastCancerAtlasRaw()
pred = pd.read_csv(f_lab)

ad.obs['is_epithelial'] = pred.clf_is_epithelial.values

# %%
markers = ['Vol7', 'H3K27me3', 'K5', 'PTEN', 'CD44', 'K8K18', 'cMYC', 'SMA', 'CD24', 'HER2', 'AR', 'BCL2', 'p53',
           'EpCAM', 'CyclinB1',
           'PRB', 'CD49f', 'Survivin', 'EZH2', 'Vimentin', 'cMET', 'AKT', 'ERa', 'CA9', 'ECadherin', 'Ki67', 'EGFR',
           'K14', 'HLADR', 'K7', 'panK']
patients = ['N_BB013',
            'N_BB028',
            'N_BB034',
            'N_BB035',
            'N_BB037',
            'N_BB046',
            'N_BB051',
            'N_BB055',
            'N_BB058',
            'N_BB064',
            'N_BB065',
            'N_BB072',
            'N_BB073',
            'N_BB075',
            'N_BB076',
            'N_BB090',
            'N_BB091',
            'N_BB093',
            'N_BB094',
            'N_BB096',
            'N_BB099',
            'N_BB101',
            'N_BB102',
            'N_BB110',
            'N_BB120',
            'N_BB131',
            'N_BB144',
            'N_BB147',
            'N_BB153',
            'N_BB154',
            'N_BB155',
            'N_BB167',
            'N_BB192',
            'N_BB194',
            'N_BB197',
            'N_BB201',
            'N_BB204',
            'N_BB209',
            'N_BB210',
            'N_BB214',
            'N_BB221']
patients = [i.split('N_')[1] for i in patients]
# %%
mask = []
for m in markers:
    mask.append(ad.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad.var['used_in_abnormality'] = mask
ad = ad[:, ad.var.used_in_abnormality]

# %%
drop_mask = []
for m in ['Vol7', 'H3K27me3', 'CyclinB1', 'Ki67']:
    drop_mask.append(ad.var.desc.str.contains(m))
drop_mask = pd.concat(drop_mask, axis=1)
drop_mask = drop_mask.any(1)

ad = ad[:, ~drop_mask]

# %% subset
ad_sub = ad[(ad.obs.is_epithelial == 1) & (ad.obs.patient_number.isin(patients)) & (ad.obs.tissue_type == 'N')]

# %% pre-processing
X = ad_sub.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

# min-max normalisation
minMax = MinMaxScaler()
X = minMax.fit_transform(X)

layer_name = 'arcsinh_norm'
ad_sub.layers[layer_name] = X


# %%
def get_autoencoder(d1, d2=10, d3=2):
    input_data = Input(shape=(d1,))
    encoded1 = Dense(d2, activation='relu')(input_data)
    encoded2 = Dense(d3, activation='relu')(encoded1)
    decoded1 = Dense(d2, activation='relu')(encoded2)
    decoded2 = Dense(d1, activation='sigmoid')(decoded1)
    ae = Model(input_data, decoded2)
    ae.compile(optimizer='adam', loss='mean_squared_error')
    return ae


ae = get_autoencoder(ad_sub.shape[1])
early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
x_train, x_test = train_test_split(ad_sub.layers[layer_name], test_size=.2)
x_train, x_val = train_test_split(ad_sub.layers[layer_name], test_size=(1 - 7 / 8))

ae.fit(x_train, x_train,
       epochs=500, verbose=1,
       batch_size=256,
       shuffle=True,
       validation_data=(x_val, x_val),
       callbacks=[early_stopping_monitor])
ae.evaluate(x_test, x_test)

# %% subset
ad_pred = ad[(ad.obs.is_epithelial == 1)]

# %% pre-processing
X = ad_pred.X.copy()

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

# min-max normalisation
X = minMax.transform(X)

layer_name = 'arcsinh_norm'
ad_pred.layers[layer_name] = X

# %%
y = ad_pred.layers[layer_name]
yhat = ae.predict(y)

mse = ((yhat - y) ** 2).mean(1)
ad_pred.obs['abnormality'] = mse
# %%
dat = ad_pred.obs.groupby(['tissue_type', 'breast', 'patient_number']).abnormality.agg(np.median).dropna().reset_index()

fig, ax = plt.subplots()
sns.boxplot(data=dat, x='tissue_type', y='abnormality', ax=ax, whis=[0, 100])
sns.stripplot(data=dat, x='tissue_type', y='abnormality', ax=ax)
fig.show()

# %%
yhat = ae.predict(ad_sub.layers[layer_name])

mse = (yhat ** 2).mean(1)
obs = ad_sub.obs.copy()
obs['abnormality'] = mse
dat = obs.groupby(['tissue_type', 'breast', 'patient_number']).abnormality.agg(np.median).dropna().reset_index()
