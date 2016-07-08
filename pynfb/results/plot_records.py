import matplotlib.pyplot as plt
import numpy as np
from pynfb.io.hdf5 import load_h5py_all_samples
import os
import glob

dir_name = max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime)
print(dir_name)
f = plt.figure()
ax = f.add_subplot(211)
ax.plot(load_h5py_all_samples(dir_name+'raw.h5')[:, 0])
ax = f.add_subplot(212)
ax.plot(load_h5py_all_samples(dir_name+'signals.h5'))
plt.show()