import numpy as np
import pylab as plt
import seaborn as sns
from scipy.stats import linregress
cm = sns.color_palette()

pps = np.load('pps.npy')

print(pps.shape)
slopes = np.zeros((20, 20))
p_values = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        res = linregress(np.arange(5), pps[:, i, j])
        slopes[i, j] = res.slope
        p_values[i, j] = res.pvalue


print(p_values)
a = sns.heatmap(slopes, mask=p_values>0.05, cmap='nipy_spectral')
#a.invert_yaxis()
#sns.heatmap(p_values * np.sign(slopes))
plt.show()