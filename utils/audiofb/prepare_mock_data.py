from utils.load_results import load_data
import pylab as plt
import numpy as np

from_file = r'C:\Users\CBI\Desktop\audio_nfb\results\fb_0_17_09-21_15-48-17\experiment_data.h5'
mock_signal_file = r'C:\Users\CBI\Desktop\audio_nfb\mock_signal.npy'


df, fs, channels, p_names = load_data(from_file)
df_fb = df.query('block_name == "FB"')[['signal_Signal', 'block_number']]
unique_fb_blocks = np.random.permutation(df_fb['block_number'].unique())
fb_signal = np.concatenate([df_fb.query(f'block_number == {k}')['signal_Signal'].values for k in unique_fb_blocks])

bl_signal = df.query('block_name == "BL" & block_number < 10')['signal_Signal']
fb_signal = (fb_signal - bl_signal.mean())/bl_signal.std()
np.save(mock_signal_file, fb_signal)

fb_signal = np.load(mock_signal_file)
plt.plot(fb_signal)
