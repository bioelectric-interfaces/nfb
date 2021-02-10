import numpy as np
import h5py
from ..signals import DerivedSignal, CompositeSignal, BCISignal


def save_h5py(file_path, data, dataset_name='dataset'):
    with h5py.File(file_path, 'a') as f:
        f.create_dataset(dataset_name, data=data)
    pass


def load_h5py(file_path, dataset_name='dataset'):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

def load_h5py_protocols_raw(file_path, protocol_indxs=None):
    if protocol_indxs is None:
        return None
    data = []
    with h5py.File(file_path, 'r') as f:
        for j in protocol_indxs:
            data.append(f['protocol{}/raw_data'.format(j + 1)][:])
    return data

def load_h5py_protocol_signals(file_path, protocol_name='protocol1'):
    with h5py.File(file_path, 'r') as f:
        if isinstance(f[protocol_name], h5py.Dataset):
            data = f[protocol_name][:]
        else:
            data = f['{}/signals_data'.format(protocol_name)][:]
    return data


def load_h5py_all_samples(file_path, raw=True):
    with h5py.File(file_path, 'r') as f:
        if isinstance(f['protocol1'], h5py.Dataset):
            data = [f['protocol' + str(j + 1)][:] for j in range(len(f.keys()))]
        else:
            keys = ['protocol{}/{}_data'.format(j + 1, 'raw' if raw else 'signals') for  j in range(len(f.keys()) - 1)]
            data = [f[key][:] for key in keys if key in f]
    return np.vstack(data)


def save_channels_and_fs(file_path, channels, fs):
    # save channels names and sampling frequency
    with h5py.File(file_path, 'a') as f:
        f.create_dataset('channels', data=np.array(channels, dtype='S'))
        f.create_dataset('fs', data=np.array(fs))

def load_channels_and_fs(file_path):
    # save channels names and sampling frequency
    with h5py.File(file_path, 'r') as f:
        return [s.decode('utf-8') for s in f['channels'][:]], int(f['fs'][()])


def save_signals(file_path, signals, group_name='protocol0', raw_data=None, timestamp_data=None, signals_data=None,
                 raw_other_data=None, reward_data=None, protocol_name='unknown', mock_previous=0, mark_data=None):
    print('Signals stats saving', group_name)
    with h5py.File(file_path, 'a') as f:
        main_group = f.create_group(group_name)
        main_group.attrs['name'] = protocol_name
        main_group.attrs['mock_previous'] = mock_previous
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
            elif isinstance(signal, BCISignal):
                signal_group.attrs['type'] = u'bci'
            else:
                raise TypeError ('Bad signal type')
            signal_group.create_dataset('mean', data=np.array(signal.mean))
            signal_group.create_dataset('std', data=np.array(signal.std))
        if raw_data is not None:
            main_group.create_dataset('raw_data', data=raw_data, compression="gzip")
        if timestamp_data is not None:
            main_group.create_dataset('timestamp_data', data=timestamp_data, compression="gzip")
        if signals_data is not None:
            main_group.create_dataset('signals_data', data=signals_data, compression="gzip")
        if raw_other_data is not None:
            main_group.create_dataset('raw_other_data', data=raw_other_data, compression="gzip")
        if reward_data is not None:
            main_group.create_dataset('reward_data', data=reward_data, compression="gzip")
        if mark_data is not None:
            main_group.create_dataset('mark_data', data=mark_data, compression="gzip")

    pass


def save_xml_str_to_hdf5_dataset(file_path, xml='', dataset_name='something.xml'):
    # Write the xml file...
    with h5py.File(file_path, 'a') as f:
        str_type = h5py.vlen_dtype(str)
        ds = f.create_dataset(dataset_name, shape=(2,), dtype=str_type)
        ds[:] = xml

def load_xml_str_from_hdf5_dataset(file_path, dataset_name='something.xml'):
    with h5py.File(file_path, 'r') as f:
        if dataset_name in f:
            xml_ = f[dataset_name][0]
            return xml_
        else:
            raise DatasetNotFound('Dataset "{}" not found in file "{}"'.format(dataset_name, file_path))


class DatasetNotFound(Exception):
    pass




if __name__ == '__main__':
    save_xml_str_to_hdf5_dataset('test.h5', 'asf1', '1st')
    save_xml_str_to_hdf5_dataset('test.h5', 'asf2', '2nd')
    with h5py.File('test.h5', 'r') as f:
        print(list(f.keys()))

