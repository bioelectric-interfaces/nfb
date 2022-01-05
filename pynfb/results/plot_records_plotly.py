import matplotlib.pyplot as plt
import numpy as np
from pynfb.serializers.hdf5 import load_h5py_all_samples, load_h5py_protocol_signals, load_h5py_protocols_raw, load_h5py
from utils.load_results import load_data
import os
import glob
import pandas as pd
import plotly.express as px
import mne

dir_name = max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime)
#dir_name = 'C:\\Users\\Nikolai\\Downloads\\composite_res\\'
print(dir_name)
# ax = f.add_subplot(211)
# ch = 20
# h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/1-nfb-task_999_12-31_14-59-31/experiment_data.h5"
h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/1-nfb-task_999_01-05_18-30-17/experiment_data.h5"
# ax.plot(load_h5py_all_samples(h5file))
# ax.set_ylabel('Raw{}'.format(ch))
# ax = f.add_subplot(212, sharex=ax)
# ax.plot(load_h5py_all_samples(h5file, raw=False))
# ax.set_ylabel('Signals')
# plt.show()

# mock_signals_buffer = load_h5py_protocol_signals(
#     h5file,
#     'protocol3')

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

# Extract the individual protocols
protocol_data = {}
block_numbers = df1['block_number'].unique()
protocol_names = [f"{a_}{b_}"for a_, b_ in zip(p_names, block_numbers)]
channels_signal = channels.copy()
channels_signal.append("signal_AAI")
df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample', 'choice', 'answer'], value_vars=channels_signal, var_name="channel", value_name='data')

for protocol_n in block_numbers:
    protocol_data[protocol_names[protocol_n-1]] = df2.loc[df2['block_number'] == protocol_n]

protocol_data['baseline2'].loc[protocol_data['baseline2']['channel'].isin(["signal_AAI"])]['data']

for protocol, data in protocol_data.items():
    p_time = (len(data)/66)/1000
    print(f"{protocol} time: {p_time}")



# fig = px.line(protocol_data["NFB5"], x="sample", y="data", color='channel')
# fig.show()

fig2 = px.line(df1, x="sample", y="signal_AAI", color='block_name')
fig2.show()
pass

# Try plot through MNE
m_info = mne.create_info(channels, fs, ch_types='eeg', verbose=None)
channel_data = df1.drop(columns=['signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'events', 'block_name', 'block_number', 'sample'])
m_raw = mne.io.RawArray(channel_data.T, m_info, first_samp=0, copy='auto', verbose=None)
m_raw.plot()
pass
