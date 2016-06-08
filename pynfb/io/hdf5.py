import numpy as np
import h5py


def save_h5py(file_path, data, dataset_name='dataset'):
    with h5py.File(file_path, 'a') as f:
        f.create_dataset(dataset_name, data=data)
    pass


def load_h5py(file_path, dataset_name='dataset'):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

def load_h5py_all_samples(file_path):
    with h5py.File(file_path, 'r') as f:
        data =[f['protocol' + str(j+1)][:] for j in range(len(f.keys()))]
    return np.vstack(data)

if __name__ == '__main__':
    a = np.random.random(size=(300, 30))
    save_h5py('temp.h5', a, 'a')
    a1 = load_h5py('temp.h5', 'a')
    print(np.allclose(a, a1))

    c = np.linspace(0, 1, 3)
    save_h5py('temp.h5', c, 'c')
    c1 = load_h5py('temp.h5', 'c')
    print(np.allclose(c, c1))

    a1 = load_h5py('temp.h5', 'a')
    print(np.allclose(a, a1))
