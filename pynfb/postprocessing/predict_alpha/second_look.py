from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import numpy as np
import pylab as plt
from pynfb.postprocessing.utils import band_hilbert
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from pynfb.signal_processing.filters import ButterBandEnvelopeDetector, ExponentialSmoother
import pandas as pd

cm = sns.color_palette()



channels = ['F1', 'F2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
fs = 500
day = 1
probe = 1
data_dir = '/home/nikolai/_Work/predict_alpha/!each data/Shamaeva/day{}/'.format(day)
file_name = 'proba{}'.format(probe)

raw = loadmat(data_dir + file_name)['X1Topo'].T
p4 = raw[:, channels.index('P4')]
t = np.arange(len(p4)) / fs

exp_sm = ExponentialSmoother(0.95)
env_detector = ButterBandEnvelopeDetector([8, 12], fs, exp_sm, 3)

#plt.plot(env_detector.apply(p4))
#plt.plot(np.abs(band_hilbert(p4, fs, (8,12))))
#plt.plot(p4*0.1)
#plt.show()


env = pd.Series(env_detector.apply(p4), index=t)
plt.plot(env)

n_samples = 1
x = env.rolling(n_samples).mean()
y = x.shift(-n_samples)

plt.plot(x)
plt.plot(y)
plt.show()

n_clusters = 7
qs = np.linspace(0, 100, n_clusters+1)
th = [np.percentile(x.dropna(), q) for q in qs]
cluster_th = [(th[k], th[k+1]) for k in range(n_clusters)]

print(cluster_th)
pp = np.zeros((n_clusters, n_clusters))

for j in range(n_clusters):
    for k in range(n_clusters):
        x0, x1 = cluster_th[j]
        y0, y1 = cluster_th[k]
        yy = y[(x >= x0) & (x < x1)]
        pp[j, k] = sum((yy >= y0) & (yy < y1))/len(yy)

f, axes = plt.subplots(1, 2)

sns.heatmap(pp, vmin=0, vmax=1, ax=axes[0])
axes[0].invert_yaxis()

#x = np.log(x)
#y = np.log(y)

axes[1].scatter(x, y, marker='.', alpha=0.5, s=1)
axes[1].vlines(th,[y.min()]*len(th), [y.max()]*len(th) )
axes[1].hlines(th,[x.min()]*len(th), [x.max()]*len(th) )
axes[1].set_xlim(x.min(), x.max())
axes[1].set_ylim(y.min(), y.max())
plt.show()

