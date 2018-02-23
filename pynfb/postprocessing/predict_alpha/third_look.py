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

band_names = ['theta', 'alpha', 'low-beta', 'beta', 'high-beta']
bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
band_dict = dict(zip(band_names, bands))
bands_to_analyse = band_names[:3]


env = pd.DataFrame(columns=bands_to_analyse, index=t)
for band in bands_to_analyse:
    exp_sm = ExponentialSmoother(0.99)
    env_detector = ButterBandEnvelopeDetector(band_dict[band], fs, exp_sm, 3)
    env[band] = env_detector.apply(p4)
#env.plot()
#plt.show()

n_clusters = 20
kmeans = KMeans(n_clusters)
kmeans.fit(env)
env['cluster'] = kmeans.predict(env)
cluster_rank = env.groupby('cluster')['alpha'].mean().rank()
print(cluster_rank)
print(cluster_rank[0])
env['rank'] = env['cluster'].apply(lambda x: int(cluster_rank[x]))

plt.plot(env['rank'])
plt.plot(env['alpha']/env['alpha'].std())
plt.show()

a = sns.pairplot(env, hue='rank', vars=bands_to_analyse, plot_kws=dict(s=1, edgecolor=None, alpha=1), palette='nipy_spectral')
plt.show()
#plt.plot(env_detector.apply(p4))
#plt.plot(np.abs(band_hilbert(p4, fs, (8,12))))
#plt.plot(p4*0.1)
#plt.show()

x = env['rank']
y = env['rank'].shift(-100)
plt.plot(x, y)
plt.show()
pp = np.zeros((n_clusters, n_clusters))

for j, rank_x in enumerate(x.dropna().unique()):
    print(rank_x)
    for k, rank_y in enumerate(y.dropna().unique()):
        pp[j, k] = sum(y[x==rank_x]==rank_y)/sum(x==rank_x)

print(pp)
f, axes = plt.subplots(1, 2)
mask = np.zeros_like(pp)
#mask[np.triu_indices_from(mask)] = 1
sns.heatmap(pp, mask=mask, ax=axes[0])
axes[0].invert_yaxis()

#x = np.log(x)
#y = np.log(y)

axes[1].scatter(x, y, alpha=0.1, s=1)
axes[1].set_xlim(x.min(), x.max())
axes[1].set_ylim(y.min(), y.max())
plt.show()

