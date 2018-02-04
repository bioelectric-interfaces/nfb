import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2, band_hilbert, get_main_band, get_main_freq
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt
from scipy.signal import welch
from scipy import fftpack
import seaborn as sns
cm = sns.color_palette()


wdir = '~/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments['name'].unique())
print(experiments)


for n_exp in range(len(experiments)):

    exp = experiments.iloc[n_exp]
    desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'exp-control', 10:'exp-control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
    print(exp, '\n*******************', desc, '\n*******************')
    df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))

    right = np.load(desc + '-RIGHT.npy')
    #left = np.load(desc + '-LEFT.npy')
    x = np.dot(df.loc[df['block_number']==3, channels], right[0])
    #plt.plot(x)
    #plt.show()


    def get_hists(x):
        nperseg = fs//2
        w = fftpack.fftfreq(nperseg, 1/fs)
        band_mask = (w < 35) & (w > 3)
        specs = []
        for k in range(0, len(x)-nperseg, nperseg):

            specs.append(2*fftpack.fft(x[k:k+nperseg])[band_mask])
        specs = np.abs(specs)

        hists = []
        bins = np.linspace(0, 0.008, 10)
        for j, f in enumerate(w[band_mask]):
            h = np.histogram(specs[:, j], bins)[0]/specs.shape[0]
            #h/=h.max()
            hists.append(h)
            #dd.append({'f':1, 'p':bins, 'v': h})
        hists = np.array(hists).T
        return hists, w[band_mask], bins


    hists, w, bins = get_hists(np.dot(df.loc[df['block_number'].isin([7, 9]), channels], right[0]))
    hists2, w, bins = get_hists(np.dot(df.loc[df['block_number'].isin([3, 1]), channels], right[0]))

    plt.figure()
    ax = sns.heatmap(hists-hists2, xticklabels=w, yticklabels=bins, cmap='BrBG', center=0, vmin=-0.05, vmax=0.05)
    ax.invert_yaxis()
    plt.yticks([])
    plt.savefig('dif/{}.png'.format(desc))
    #plt.plot(w[band_mask], specs.mean(0))
    #plt.show()