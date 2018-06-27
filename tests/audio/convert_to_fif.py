import numpy as np
import os
import mne


def save_fif(record_dir):
    with open('{}\montage.info'.format(record_dir), 'r') as f:
        fs, *channels = f.readline().split(' ')
        fs = int(fs)

    records = []
    for record in os.listdir(record_dir):
        if '.npy' in record:
            records.append(np.load('{}/{}'.format(record_dir, record)))

    records = np.concatenate(records)
    montage = mne.channels.read_montage('standard_1005')
    upper_ch_names = [ch.upper() for ch in montage.ch_names]
    ch_types = [('eeg' if ch.upper().split('-')[0] in upper_ch_names else 'misc') for ch in channels]
    print(list(zip(channels, ch_types)))
    info = mne.create_info(channels, fs, ch_types, montage=montage)
    raw = mne.io.RawArray(records.T, info)
    raw.save('{0}/{0}_raw.fif'.format(record_dir))

if __name__ == '__main__':
    save_fif('test_med')