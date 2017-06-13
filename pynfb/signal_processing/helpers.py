import numpy as np


def get_outliers_mask(data_raw: np.ndarray, iter_numb=10, std=4):
    data_pwr = np.sqrt(np.sum(data_raw ** 2, 1))
    indexes = np.arange(data_pwr.shape[0])
    for i in range(iter_numb):
        mask = data_pwr - data_pwr.mean() < std * data_pwr.std()
        indexes = indexes[mask]
        data_pwr = data_pwr[mask]
    print('Dropped {} outliers'.format(data_raw.shape[0] - len(indexes)))
    outliers_mask = np.ones(shape=(data_raw.shape[0], ))
    outliers_mask[indexes] = 0
    return outliers_mask.astype(bool)

if __name__ == '__main__':
    import pylab as plt
    data = np.random.normal(size=(10000, 20))
    outliers_indexes = np.random.randint(0, 10000, 10)
    print(sorted(outliers_indexes))
    data[outliers_indexes] *= 2
    good_mask = get_outliers_mask(data)
    print(sorted([k for k in range(10000) if k not in good_mask]))
    plt.plot(np.sqrt(np.sum(data ** 2, 1)))
    plt.hlines(4*np.sqrt(np.sum(data ** 2, 1)).std()+np.sqrt(np.sum(data ** 2, 1)).mean(), 0, 10000, )
    plt.show()