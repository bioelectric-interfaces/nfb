import matplotlib.pyplot as plt
import numpy as np
from pynfb.serializers.hdf5 import load_h5py_all_samples, load_h5py_protocol_signals, load_h5py_protocols_raw, load_h5py
from utils.load_results import load_data
import os
import glob
import pandas as pd
import plotly.express as px
import mne
from pynfb.signal_processing.filters import ExponentialSmoother, SGSmoother, FFTBandEnvelopeDetector, \
    ComplexDemodulationBandEnvelopeDetector, ButterBandEnvelopeDetector, ScalarButterFilter, IdentityFilter, \
    FilterSequence, DelayFilter, CFIRBandEnvelopeDetector

dir_name = max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime)
#dir_name = 'C:\\Users\\Nikolai\\Downloads\\composite_res\\'
print(dir_name)
# ax = f.add_subplot(211)
# ch = 20
# h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/0-nfb_task_tst_01-23_14-16-33/experiment_data.h5"
# h5file = "/Users/christopherturner/Documents/EEG_Data/pilot_202201/ct/scalp/0-nfb_task_ct01_lab_01-10_16-27-25/experiment_data.h5" #CHRIS PILOT
# h5file = '/Users/christopherturner/Documents/EEG_Data/pilot_202201/sh/scalp/0-nfb_task_SH01_01-11_15-50-56/experiment_data.h5' # SIMON PILOT
h5file = '/Users/christopherturner/Documents/EEG_Data/pilot_202201/kk/scalp/0-nfb_task_kk_01-21_20-41-19/experiment_data.h5' # Ksenia PILOT

h5file = f"/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/1-nfb_task_cvsa_test_04-07_10-17-39/experiment_data.h5"

# LSL MARKER TEST
h5file = "/Users/christopherturner/Documents/EEG_Data/aai_testing_20220601/0-nfb_task_PO0_1_06-01_17-27-12/experiment_data.h5"
# h5file = "/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/lsl_markers_test_06-03_00-19-50/experiment_data.h5"
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
# df1.signal_AAI.to_pickle('/Users/christopherturner/Documents/GitHub/nfb/pynfb/results/aai.pkl')

probes = df1.loc[df1.probe.isin([1,2])]
block_transitions = df1.loc[df1['block_number'].diff() == 1]
inspection = df1.loc[df1.block_number == 15]

aai_derived_sigs = (abs(df1.signal_Alpha_Left) - abs(df1.signal_Alpha_Right)) / (abs(df1.signal_Alpha_Left) + abs(df1.signal_Alpha_Right))

aai_crop = df1.iloc[0:10000,:]
fig = px.line(aai_crop, x=aai_crop.index, y="signal_AAI")
# fig.add_scatter(x=df1.index, y=df1["signal_Alpha_Right"])
# fig.add_scatter(x=df1.index, y=df1["signal_AAI"])
# fig.add_scatter(x=aai_derived_sigs.index, y=aai_derived_sigs)
fig.show()

m_info = mne.create_info(channels, fs, ch_types='eeg', verbose=None)
channel_data = df1.drop(columns=['signal_Alpha_Left', 'signal_Alpha_Right', 'signal_AAI', 'events', 'reward', 'choice', 'answer', 'probe', 'chunk_n', 'block_name', 'block_number', 'sample'])

left_chs = 'C1'
left_data = channel_data.drop(columns=[x for x in channel_data.columns if x != left_chs])
signal_estimator = FFTBandEnvelopeDetector((8, 12), fs, SGSmoother(151, 2), 250)
current_chunk = signal_estimator.apply(left_data.to_numpy())

# (abs(Alpha_Left)-abs(Alpha_Right))/(abs(Alpha_Left)+abs(Alpha_Right))


m_raw = mne.io.RawArray(channel_data.T, m_info, first_samp=0, copy='auto', verbose=None)

drops = [x for x in m_info.ch_names if x != 'C1']
m_raw.drop_channels(drops)

# Extract the individual protocols
protocol_data = {}
block_numbers = df1['block_number'].unique()
protocol_names = [f"{a_}{b_}"for a_, b_ in zip(p_names, block_numbers)]
channels_signal = channels.copy()
# channels_signal.append("signal_AAI")
df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample', 'choice', 'answer', 'probe'], value_vars=channels_signal, var_name="channel", value_name='data')

for protocol_n in block_numbers:
    protocol_data[protocol_names[protocol_n-1]] = df2.loc[df2['block_number'] == protocol_n]

# protocol_data['baseline2'].loc[protocol_data['baseline2']['channel'].isin(["signal_AAI"])]['data']


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
