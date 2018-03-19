import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt

wdir = '/home/nikolai/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)


n_exp = 24
print(n_exp)


exp = experiments.iloc[n_exp]
desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
print(exp, '\n*******************', desc, '\n*******************')
df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))


from scipy.io import savemat


print(df.head())
df = df[~get_outliers_mask(df[channels], std=3)]

right, left = runica2(df.loc[df['block_number'].isin([1, 2, 3, 7, 8, 9]), channels], fs, channels, ['RIGHT', 'LEFT'])
np.save(wdir + desc + '-RIGHT.npy', right)
np.save(wdir + desc + '-LEFT.npy', left)


# load and plot
right = np.load(wdir + desc + '-RIGHT.npy')
left = np.load(wdir + desc + '-LEFT.npy')
f, ax = plt.subplots(1, 4)
plot_topomap(right[0], Montage(channels).get_pos(), contours=0, axes=ax[0], show=False)
plot_topomap(right[1], Montage(channels).get_pos(), contours=0, axes=ax[1], show=False)
plot_topomap(left[0], Montage(channels).get_pos(), contours=0, axes=ax[2], show=False)
plot_topomap(left[1], Montage(channels).get_pos(), contours=0, axes=ax[3], show=False)
ax[0].set_title('Right spat.')
ax[1].set_title('Right topog.')
ax[2].set_title('Left spat.')
ax[3].set_title('Left topog.')
ax[0].set_ylabel(desc)
plt.savefig(wdir + desc+'-SMR-filters.png')
plt.show()