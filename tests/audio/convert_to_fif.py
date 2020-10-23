import numpy as np
import os
import mne

names_map = {'Po3': 'Fc5',
             'Oz' : 'Fc6',
             'Po4': 'Oz',
             'Fc1': 'Po4',
             'Fc2': 'Po3',
             'Fc6': 'Fc2',
             'Fc5': 'Fc1'}


def save_fif(record_dir, fix_names=False):
    with open('{}\montage.info'.format(record_dir), 'r', encoding="utf-8") as f:
        fs, *channels = f.readline().split(' ')
        fs = int(fs)

    if fix_names:
        channels = [(names_map[ch] if ch in names_map else ch) for ch in channels]

    records = []
    n_files = sum([('.npy' in record) for record in os.listdir(record_dir)])
    for r in range(n_files):
        records.append(np.load('{}/record_{}.npy'.format(record_dir, r+1)))

    records = np.concatenate(records)
    montage = mne.channels.read_montage('standard_1005')
    upper_ch_names = [ch.upper() for ch in montage.ch_names]

    ch_types = [('eeg' if ch.upper().split('-')[0] in upper_ch_names else 'misc') for ch in channels]
    print(list(zip(channels, ch_types)))
    info = mne.create_info(channels, fs, ch_types, montage=montage)
    raw = mne.io.RawArray(records.T, info)
    raw.save('{0}/{0}_raw.fif'.format(record_dir))

if __name__ == '__main__':
    save_fif('ni-sm')