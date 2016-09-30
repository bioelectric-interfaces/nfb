import numpy as np
import h5py
from pynfb.signals import DerivedSignal, CompositeSignal


def save_h5py(file_path, data, dataset_name='dataset'):
    with h5py.File(file_path, 'a') as f:
        f.create_dataset(dataset_name, data=data)
    pass


def load_h5py(file_path, dataset_name='dataset'):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

def load_h5py_protocol_signals(file_path, protocol_name='protocol1'):
    with h5py.File(file_path, 'r') as f:
        if isinstance(f[protocol_name], h5py.Dataset):
            data = f[protocol_name][:]
        else:
            data = f['{}/signals_data'.format(protocol_name)][:]
    return data


def load_h5py_all_samples(file_path):
    with h5py.File(file_path, 'r') as f:
        if isinstance(f['protocol1'], h5py.Dataset):
            data = [f['protocol' + str(j + 1)][:] for j in range(len(f.keys()))]
        else:
            data = [f['protocol{}/raw_data'.format(j + 1)][:] for j in range(len(f.keys()) - 1)]
    return np.vstack(data)


def save_signals(file_path, signals, group_name='protocol0', raw_data=None, signals_data=None, raw_other_data=None,
                 reward_data=None):
    print('Signals stats saving', group_name)
    with h5py.File(file_path, 'a') as f:
        main_group = f.create_group(group_name)
        signals_group = main_group.create_group('signals_stats')
        for signal in signals:
            signal_group = signals_group.create_group(signal.name)
            if isinstance(signal, DerivedSignal):
                signal_group.attrs['type'] = u'derived'
                rejections_group = signal_group.create_group('rejections')
                for k, rejection in enumerate(signal.rejections.list):
                    dataset = rejections_group.create_dataset('rejection'+str(k+1), data=np.array(rejection.val))
                    rejections_group.create_dataset('rejection' + str(k + 1) + '_topographies',
                                                    data=np.array(rejection.topographies))
                    dataset.attrs['type'] = rejection.type_str
                    dataset.attrs['rank'] = rejection.rank
                signal_group.create_dataset('spatial_filter', data=np.array(signal.spatial_filter))
                signal_group.create_dataset('bandpass', data=np.array(signal.bandpass))
            elif isinstance(signal, CompositeSignal):
                signal_group.attrs['type'] = u'composite'
            else:
                raise TypeError ('Bad signal type')
            signal_group.create_dataset('mean', data=np.array(signal.mean))
            signal_group.create_dataset('std', data=np.array(signal.std))
        if raw_data is not None:
            main_group.create_dataset('raw_data', data=raw_data)
        if signals_data is not None:
            main_group.create_dataset('signals_data', data=signals_data)
        if raw_other_data is not None:
            main_group.create_dataset('raw_other_data', data=raw_other_data)
        if reward_data is not None:
            main_group.create_dataset('reward_data', data=reward_data)
    pass
if __name__ == '__main__':
    load_h5py('C:\\Users\\Nikolai\PycharmProjects\pynfb_repo\\nfb\pynfb\\results\MU_test_AN_09-29_17-15-38'
                          '\experiment_data.h5')
