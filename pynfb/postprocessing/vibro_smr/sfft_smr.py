import pandas as pd
from pynfb.postprocessing.utils import load_data, runica2
from pynfb.signal_processing.helpers import get_outliers_mask
import numpy as np
from pynfb.inlets.montage import Montage
from mne.viz import plot_topomap
import pylab as plt
from scipy import signal

wdir = '/home/nikolai/_Work/vibro_smr/'
data_dir = '/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/Desktop/vibro-decay/'
experiments = pd.read_csv(wdir + 'vibro-decay.csv')
experiments = experiments[experiments.protocol == 'belt']
print(experiments)

for n_exp in range(len(experiments)):
    print(n_exp)


    exp = experiments.iloc[n_exp]
    desc = '{}-{}-{}-{}'.format(exp['subject'], exp['protocol'], {0: 'exp', 1:'control'}[exp['control']], '-'.join(exp.dataset.split('_')[-2:]))
    print(exp, '\n*******************', desc, '\n*******************')
    df, fs, p_names, channels = load_data('{}{}/experiment_data.h5'.format(data_dir, exp.dataset))
    # df = df[~get_outliers_mask(df[channels], std=3)]


    right = np.load(wdir + desc + '-RIGHT.npy')[0]
    left = np.load(wdir + desc + '-LEFT.npy')[0]

    x = np.dot(df[channels], right)


    fig, axes = plt.subplots(2, 1, sharex=True)

    t = np.arange(len(x))/fs/60
    axes[0].plot(t, x*1000*1000)
    axes[0].set_ylabel('Voltage [$\mu$V]')
    #axes[0].set_ylim(-max(x), 500)

    f, t, Sxx = signal.spectrogram(x, fs, scaling='spectrum', nfft=fs*2)
    ax = axes[1].pcolormesh(t/60, f, np.log10(Sxx**0.5), vmin=-7, vmax=-4.8, cmap='nipy_spectral')
    axes[1].set_ylabel('Frequency [Hz]')
    axes[1].set_xlabel('Time [min]')
    axes[0].set_xlim(0, t.max()/60)
    axes[1].set_ylim(0, 75)
    axes[0].set_title(exp['name'] +'-LEFT-' + desc)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.35])
    cb = fig.colorbar(ax, cax=cbar_ax, ticks=[ -5, -6, -7, -8])
    cbar_ax.set_ylabel('Log magnitude [logV]')

    plt.savefig(wdir+desc+'-left-specgram.png', dpi=200)
    plt.close()
    #plt.show()