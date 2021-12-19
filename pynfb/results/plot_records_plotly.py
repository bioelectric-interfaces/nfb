import matplotlib.pyplot as plt
import numpy as np
from pynfb.serializers.hdf5 import load_h5py_all_samples, load_h5py_protocol_signals, load_h5py_protocols_raw, load_h5py
from utils.load_results import load_data
import os
import glob
import pandas as pd
import plotly.express as px

dir_name = max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime)
#dir_name = 'C:\\Users\\Nikolai\\Downloads\\composite_res\\'
print(dir_name)
# ax = f.add_subplot(211)
# ch = 20
h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/alpha_synch_scalp_12-19_14-08-09/experiment_data.h5"
# ax.plot(load_h5py_all_samples(h5file))
# ax.set_ylabel('Raw{}'.format(ch))
# ax = f.add_subplot(212, sharex=ax)
# ax.plot(load_h5py_all_samples(h5file, raw=False))
# ax.set_ylabel('Signals')
# plt.show()

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# Extract the individual protocols
protocol_data = {}
block_numbers = df1['block_number'].unique()
protocol_names = [f"{a_}{b_}"for a_, b_ in zip(p_names, block_numbers)]
df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample'], value_vars=channels.append("signal_AAI"), var_name="channel", value_name='data')

for protocol_n in block_numbers:
    protocol_data[protocol_names[protocol_n-1]] = df2.loc[df2['block_number'] == protocol_n]

fig = px.line(protocol_data["NFB5"], x="sample", y="data", color='channel')
fig.show()
pass

