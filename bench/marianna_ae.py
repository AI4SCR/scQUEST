# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')


# %%

def train_autoencoder(x_train, x_test, d1, d2=10, d3=2):
    input_data = Input(shape=(d1,))
    encoded1 = Dense(d2, activation='relu')(input_data)
    encoded2 = Dense(d3, activation='relu')(encoded1)
    decoded1 = Dense(d2, activation='relu')(encoded2)
    decoded2 = Dense(d1, activation='sigmoid')(decoded1)
    ae = Model(input_data, decoded2)
    encoder = Model(input_data, encoded2)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(x_train, x_train,
           epochs=500, verbose=1,
           batch_size=256,
           shuffle=True,
           validation_data=(x_test, x_test),
           callbacks=[early_stopping_monitor])
    return ae


# %%
import starProtocols as sp
import pandas as pd
import numpy as np

# %%
ad = sp.dataset.breastCancerAtlasRaw()
marker = sp.utils.DEFAULT_MARKER_CLF

# %%
mask = []
for m in marker:
    mask.append(ad.var.desc.str.contains(m))
mask = pd.concat(mask, axis=1)
mask = mask.any(1)
ad.var['used_in_clf'] = mask
ad = ad[:, ad.var.used_in_clf]

# %%
drop_mask = []
for m in ['Vol7', 'H3K27me3', 'CyclinB1', 'Ki67']:
    drop_mask.append(ad.var.desc.str.contains(m))
drop_mask = pd.concat(drop_mask, axis=1)
drop_mask = drop_mask.any(1)

ad = ad[:, ~drop_mask]

# %% pre-processing
q999 = np.quantile(ad.X, .999, axis=0)

X = ad.X.copy()
for i, qmax in zip(range(X.shape[1]), q999):
    X[X[:, i] > qmax, i] = qmax

# arcsinh transform
cofactor = 5
np.divide(X, cofactor, out=X)
np.arcsinh(X, out=X)

layer_name = 'censor_arcsinh_norm'
ad.layers[layer_name] = X

# %%
ad_train = ad[ad.obs.tissue_type == 'N', :]
minMax = MinMaxScaler()

X_fit = ad_train.layers[layer_name]
X_fit = minMax.fit_transform(X_fit)

# %%
X_train, X_test = train_test_split(X_fit, test_size=.3)
ae = train_autoencoder(x_train=X_train, x_test=X_test, d1=X_train.shape[1])

# %%
X_T = ad[ad.obs.tissue_type == 'T', :].layers[layer_name]
X_N = ad[ad.obs.tissue_type == 'N', :].layers[layer_name]
X_H = ad[ad.obs.tissue_type == 'H', :].layers[layer_name]

X_T = minMax.transform(X_T)
X_N = minMax.transform(X_N)
X_H = minMax.transform(X_H)

y_T = ae.predict(X_T)
y_N = ae.predict(X_N)
y_H = ae.predict(X_H)

mse_T = ((y_T - X_T) ** 2).mean(1)
mse_N = ((y_N - X_N) ** 2).mean(1)
mse_H = ((y_H - X_H) ** 2).mean(1)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cont = []
for i, j in zip([mse_T, mse_N, mse_H], ['T', 'N', 'H']):
    p = ad[ad.obs.tissue_type == j].obs.patient_number.values
    cont.append(pd.DataFrame(i, columns=['mse']).assign(tissue=j, patient=p))

df = pd.concat(cont)
df = df.groupby(['tissue', 'patient']).mean().reset_index()

fig, ax = plt.subplots()
sns.boxplot(data=df, x='tissue', y='mse', ax=ax, whis=[0, 100])
sns.stripplot(data=df, x='tissue', y='mse', ax=ax, color=".3")
fig.tight_layout()
fig.show()
