# %%
import starProtocols as sp
from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# %%
data = sp.DummyData()
clf = sp.Classifier(n_in=data[0][0].shape[0], hidden=(3, 3, 3), n_out=data[0][1].shape[0])
print(clf)

i = data[0][0]
clf(data[0][0])
# %%
data = sp.LitDummyData()
clf = sp.Classifier(n_in=data[0][0].shape[0], hidden=(3, 3, 3), n_out=data[0][1].shape[0])
model = sp.LitClassifier(model=clf, loss_fn=nn.CrossEntropyLoss())

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, datamodule=data)
yhat = model(data[:][0])
y = data[:][1].argmax(1)
