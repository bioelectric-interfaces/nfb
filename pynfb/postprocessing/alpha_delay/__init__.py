from utils.load_results import load_data
import pylab as plt
import pandas as pd
import seaborn as sns
import numpy as np
cm = sns.color_palette()

file_path = r'/home/nikolai/Desktop/experiment_data.h5'




df, fs, channels = load_data(file_path)
print(df)
df['time'] = np.arange(len(df))
for k in df['block_number'].unique():
    df.loc[df['block_number']==k, 'time'] = df.loc[df['block_number']==k, 'time'] - df.loc[df['block_number']==k, 'time'].iloc[0]

#plt.plot(np.arange(len(df))/fs, df[ 'signal_Composite'])
#plt.plot(np.arange(len(df))/fs, df.loc[df['block_name']=='Thumb', 'signal_Composite'])
#plt.show()

sig = 'Thumb/Pinky'
states = {'Pinky': 'signal_Pi', 'Thumb': 'signal_Th', 'Thumb/Pinky': 'signal_Composite'}

df['time'] = df['time']/fs
sns.tsplot(df, time='time', value=states[sig], condition='block_name', unit='block_number', ci='sd')
plt.xlabel('Time, s')
plt.ylabel('Magnitude, V')
plt.title('Signal: ' + sig)
plt.tight_layout()
plt.show()

df.to_csv('myoexp1.csv')
