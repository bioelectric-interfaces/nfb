from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import numpy as np
import pylab as plt
from pynfb.postprocessing.utils import band_hilbert
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from pynfb.signal_processing.filters import ButterBandEnvelopeDetector, ExponentialSmoother
import pandas as pd

cm = sns.color_palette()

def load_p4_data(subj_dir):
    channels = ['F1', 'F2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    p4_ind = channels.index('P4')
    fs = 500
    files = ['fon', 'proba1', 'proba2', 'proba3', 'proba4', 'proba5']
    raw = pd.DataFrame(columns=['p4', 'block', 'block_name', 'day'])
    block = 0
    for day in [1, 2]:
        for file_name in files:
            p4 = loadmat('{}/day{}/{}.mat'.format(subj_dir, day, file_name))['X1Topo'].T[:, p4_ind]
            raw = raw.append(pd.DataFrame({'p4': p4, 'block': block, 'block_name': file_name, 'day': day}), ignore_index=1)
            block += 1
    return raw, channels, fs





band_names = ['theta', 'alpha', 'low-beta', 'beta', 'high-beta']
bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
band_dict = dict(zip(band_names, bands))
bands_to_analyse = band_names[:3]


subj_dir = '/home/nikolai/_Work/predict_alpha/!each data/Dasha'

raw, channels, fs = load_p4_data(subj_dir)
for day in []:
    scaler = RobustScaler()
    scaler.fit(raw.loc[raw.day == day, 'p4'])
    scaler.fit(raw.loc[raw.day == day, 'p4'])

for band in bands_to_analyse:
    exp_sm = ExponentialSmoother(0.99)
    env_detector = ButterBandEnvelopeDetector(band_dict[band], fs, exp_sm, 3)
    raw[band] = env_detector.apply(raw['p4'])
    for day in []:
        #raw.loc[raw.day == day, band] -= raw.loc[(raw.day == day) & (raw.block_name == 'fon'), band].quantile(0.05)
        #print('mode', raw.loc[(raw.day == day) & (raw.block_name == 'fon'), band].mode())
        raw.loc[raw.day == day, band] /= raw.loc[(raw.day == day), band].quantile(0.01)

plt.plot(raw.loc[(raw.day == 1) & (raw.block_name == 'fon'), 'p4'], label='day1')
plt.plot(raw.loc[(raw.day == 2) & (raw.block_name == 'fon'), 'p4'], label='day2')
plt.legend()
plt.show()

sns.kdeplot(raw.loc[raw.day == 1, 'alpha'])
sns.kdeplot(raw.loc[raw.day == 2, 'alpha'])
plt.show()
print(raw.head())

plt.plot(raw['alpha'])
plt.show()
