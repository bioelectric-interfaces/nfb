import matplotlib.pyplot as plt
import numpy as np
from pynfb.io.hdf5 import load_h5py_all_samples
import os
import glob

dir_name = max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime)
#dir_name = 'C:\\Users\\Nikolai\\Downloads\\composite_res\\'
print(dir_name)
f = plt.figure()
ax = f.add_subplot(211)
ch = 20
h5file = 'C:\\Users\\Nikolai\Downloads\pilot\pilot_Plackhin_1_10-20_12-03-01\experiment_data.h5'
ax.plot(load_h5py_all_samples(h5file))
ax.set_ylabel('Raw{}'.format(ch))
ax = f.add_subplot(212, sharex=ax)
ax.plot(load_h5py_all_samples(h5file, raw=False))
ax.set_ylabel('Signals')
plt.show()

