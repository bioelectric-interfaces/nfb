import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt

wdir = '~/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)


n_exp = 19


exp = experiments.iloc[n_exp]
desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'exp-control', 10:'exp-control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
print(exp, '\n*******************', desc, '\n*******************')
df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))
right, left = runica2(df.loc[df['block_number'].isin([1, 2, 3, 7, 8, 9]), channels], fs, channels, ['RIGHT', 'LEFT'])
np.save(desc + '-RIGHT.npy', right)
np.save(desc + '-LEFT.npy', left)


# load and plot
right = np.load(desc + '-RIGHT.npy')
left = np.load(desc + '-LEFT.npy')
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
plt.savefig(desc+'-SMR-filters.png')
plt.show()