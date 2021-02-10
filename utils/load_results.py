import xml.etree.ElementTree as ET
import h5py
import pandas as pd
import numpy as np


def _get_channels_and_fs(xml_str_or_file):
    root = ET.fromstring(xml_str_or_file)
    if root.find('desc').find('channels') is not None:
        channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    else:
        channels = [k.find('name').text for k in root.find('desc').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs


def _get_signals_list(xml_str):
    root = ET.fromstring(xml_str)
    derived = [s.find('sSignalName').text for s in root.find('vSignals').findall('DerivedSignal')]
    composite = []
    if root.find('vSignals').findall('CompositeSignal')[0].find('sSignalName') is not None:
        composite = [s.find('sSignalName').text for s in root.find('vSignals').findall('CompositeSignal')]
    return derived + composite


def _get_info(f):
    if 'channels' in f:
        channels = [ch.decode("utf-8")  for ch in f['channels'][:]]
        fs = int(f['fs'][()])
    else:
        channels, fs = _get_channels_and_fs(f['stream_info.xml'][0])
    signals = _get_signals_list(f['settings.xml'][0])
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    block_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    return fs, channels, block_names, signals


def load_data(file_path):
    with h5py.File(file_path) as f:
        # load meta info
        fs, channels, p_names, signals = _get_info(f)

        # load raw data
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
        df = pd.DataFrame(np.concatenate(data), columns=channels)

        # load signals data
        signals_data = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
        df_signals = pd.DataFrame(np.concatenate(signals_data), columns=['signal_'+s for s in signals])
        df = pd.concat([df, df_signals], axis=1)

        # load timestamps
        if 'timestamp' in df:
            timestamp_data = [f['protocol{}/timestamp_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['timestamps'] = np.concatenate(timestamp_data)

        # events data
        events_data = [f['protocol{}/mark_data'.format(k + 1)][:] for k in range(len(p_names))]
        df['events'] = np.concatenate(events_data)

        # set block names and numbers
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
    return df, fs, channels, p_names


if __name__ == '__main__':
    file_path = r'/media/nikolai/D27ECFCB7ECFA697/Users/Nikolai/PycharmProjects/nfb/pynfb/results/' \
                r'delay-p4_02-20_11-38-03/experiment_data.h5'
    df, fs, channels = load_data(file_path)
    print(df.groupby('block_number')['block_name'].first())
    print(df.head())