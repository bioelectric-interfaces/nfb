from pynfb.io.xml_ import xml_file_to_params
from pynfb.io.hdf5 import load_xml_str_from_hdf5_dataset
from pynfb.signals import DerivedSignal
from utils.load_results import load_data
import pylab as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import h5py


def restore_online_signal(h5_dataset, signal_name, band=None, spatial_filter=None, smoothing_factor=0.999):
    df, fs, channels, p_names = load_data(h5_dataset)
    n_channels = len(channels)

    with h5py.File(h5_dataset) as f:
        last_block = len(df.block_number.unique())
        if spatial_filter is None:
            spatial_filter = f['protocol{}/signals_stats/Signal/spatial_filter'.format(last_block)][:]
        if band is None:
            band = f['protocol{}/signals_stats/Signal/bandpass'.format(last_block)][:]

    xml_settings = load_xml_str_from_hdf5_dataset(h5_dataset, 'settings.xml')
    params = xml_file_to_params(xml_settings)
    signals_params_list = params['vSignals']['DerivedSignal']
    signals_names = [s['sSignalName'] for s in signals_params_list]
    signal = signals_params_list[signals_names.index(signal_name)]

    signal['fBandpassLowHz'], signal['fBandpassHighHz'] = band
    signal['sTemporalFilterType'] = 'butter'
    signal['fSmoothingFactor'] = smoothing_factor
    signal = DerivedSignal.from_params(0, fs, n_channels, channels, signal, spatial_filter)

    time_series = signal.update(df[channels].values)
    return signal, time_series


if __name__ == '__main__':
    h5_dataset = r'C:\Projects\nfblab\nfb\pynfb\results\S2-Loco-Real-Sit_07-20_18-35-10\experiment_data.h5'
    class_labels = ['Prepare', 'Go']

    df, fs, channels, p_names = load_data(h5_dataset)
    signal_name, time_series = restore_online_signal(h5_dataset, 'Signal', band=None, spatial_filter=None, smoothing_factor=0.99)
    df['smr'] = time_series

    classifier = LogisticRegression()
    X = df.loc[df.block_name.isin(class_labels), 'smr'].values.reshape(-1, 1)
    y = (df.loc[df.block_name.isin(class_labels), 'block_name']==class_labels[1]).astype(int).values
    classifier.fit(X, y)
    p_pred = classifier.predict_proba(X)[:, 1]
    y_pred = classifier.predict(X)
    print(sum(y_pred==y)/len(y_pred))
    print(roc_auc_score(y, p_pred))
    plt.plot(classifier.predict_proba(df['smr'].values.reshape(-1, 1))[:, 1])
    plt.plot(df['smr'].values/df['smr'].values.std())
    plt.show()