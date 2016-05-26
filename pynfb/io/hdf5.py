import numpy as np
import h5py


def save_h5py(file_path, data):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dataset', data=data)
    pass


def load_h5py(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['dataset'][:]
    return data


if __name__ == '__main__':
    a = np.random.random(size=(300, 30))
    save_h5py('temp.h5', a)
    b = load_h5py('temp.h5')
    print(np.allclose(a, b))