# %%
import starProtocols as sp

# %%
ad = sp.dummy_annData()
ab = sp.Abnormality()
ab.fit(ad, obs_col='tea', normal_id='normal')
ab.predict(ad)

# %%
