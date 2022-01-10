import matplotlib.pyplot as plt
import numpy as np
from pynfb.serializers.hdf5 import load_h5py_all_samples
import os
import glob

dir_name = max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime)
#dir_name = 'C:\\Users\\Nikolai\\Downloads\\composite_res\\'
print(dir_name)
f = plt.figure()
ax = f.add_subplot(211)
ch = 20
h5file ="/Users/2354158T/Documents/GitHub/nfb/pynfb/results/alpha_synch_scalp_12-16_11-31-26/experiment_data.h5"
ax.plot(load_h5py_all_samples(h5file))
ax.set_ylabel('Raw{}'.format(ch))
ax = f.add_subplot(212, sharex=ax)
ax.plot(load_h5py_all_samples(h5file, raw=False))
ax.set_ylabel('Signals')
plt.show()

