import h5py
import numpy as np
import pylab as plt
import pandas as pd
from scipy import signal
from matplotlib import cm

rooms = {'104': r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\ghosts-in-104_01-27_18-01-02\experiment_data.h5',
            '111': r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\ghost-rec_01-26_23-14-05\experiment_data.h5',
         '111-2':  r'/home/nikolai/_Work/alpha_delayed_nfb/p4-nfb-pilot_02-06_14-28-12/experiment_data.h5'}
room = '111-2'

with h5py.File(rooms[room]) as f:
    print(list(f.keys()), list(f['protocol1'].keys()))
    data = [f['protocol{}/raw_data'.format(k+1)][:] for k in range(len(f.keys())-3)]
fs = 500

fig, axes = plt.subplots(2, 1, sharex=True)

x = np.concatenate(data)[:, 0]
t = np.arange(len(x))/fs/60
axes[0].plot(t, x*1000*1000)
axes[0].set_ylabel('Voltage [$\mu$V]')
axes[0].set_ylim(-200, 200)
#axes[0].set_title('Room: {} Date: {} Start time: {}'.format(room.split('-')[0], *rooms[room].split('\\')[-2].split('_')[-2:]))

f, t, Sxx = signal.spectrogram(x, fs, scaling='spectrum')
ax = axes[1].pcolormesh(t/60, f, np.log10(Sxx**0.5), vmin=-7.1, vmax=-4, cmap='nipy_spectral')
axes[1].set_ylabel('Frequency [Hz]')
axes[1].set_xlabel('Time [min]')
axes[0].set_xlim(0, t.max()/60)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.35])
cb = fig.colorbar(ax, cax=cbar_ax, ticks=[-3, -4, -5, -6, -7])
cbar_ax.set_ylabel('Log magnitude [logV]')

plt.savefig('ghost{}.png'.format(room), dpi=200)
plt.show()

