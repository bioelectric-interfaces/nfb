import matplotlib.pyplot as plt
import numpy as np


f = plt.figure()
ax = f.add_subplot(211)
ax.plot(np.load('signals.npy'))
ax = f.add_subplot(212)
ax.plot(np.load('raw.npy')[:, 0])
plt.show()