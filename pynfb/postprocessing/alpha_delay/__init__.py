from pynfb.postprocessing.utils import load_data, fft_filter, dc_blocker, load_signals_data
import pylab as plt
import pandas as pd
import seaborn as sns
import numpy as np
cm = sns.color_palette()

file_path = r'/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/PycharmProjects/nfb/pynfb/results/delay-p4_02-20_11-38-03/experiment_data.h5'




df, fs, p_names, channels = load_data(file_path)
signals = load_signals_data(file_path)
print(signals)
print('*****', p_names)

data = pd.DataFrame()

data['p4'] = dc_blocker(df['P4'])
data['signal'] = signals['Signal']
data['block_name'] = df['block_name']
data['block_number'] = df['block_number']
data.to_csv('alpha-delayed-20-02-18.csv')

labels = []
handles = []
b_names = list(data['block_name'].unique())
data.index = np.arange(len(data)) / fs
for k in data['block_number'].unique():
    x = data.loc[data['block_number'] == k]
    name = x['block_name'].iloc[0]

    print(name)
    h,  = plt.plot(x['signal'].rolling(fs*30, center=True).mean(), c=cm[b_names.index(name)], label=name)
    plt.plot(x['signal'], c=cm[b_names.index(name)], alpha=0.3)
    if name not in labels:
        labels.append(name)
        handles.append(h)

    #plt.plot(x['signal'], c=cm[p_names.index(name)], alpha=0.2)

plt.legend(handles=handles)
plt.xlabel('Time [s]')
plt.ylabel('Magnitude [V]')
plt.show()