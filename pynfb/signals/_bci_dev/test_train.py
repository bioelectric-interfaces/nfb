from pynfb.signals._bci_dev.bcimodel_draft import BCIModel
from pynfb.signal_processing.filters import ButterFilter, FilterStack, InstantaneousVarianceFilter, FilterSequence
from pynfb.signal_processing.decompositions import SpatialDecompositionPool
import numpy as np
import pylab as plt
from pynfb.helpers.dc_blocker import DCBlocker

from sklearn.neural_network import MLPClassifier

filename = 'C:\\Users\\nsmetanin\\PycharmProjects\\nfb\\pynfb\\signals\\_bci_dev\\sm_ksenia_1.mat'
[eeg_data1, states_labels1, fs, chan_names, chan_numb, samp_numb, states_codes] = BCIModel.open_eeg_mat(filename, centered=False)
filename = 'C:\\Users\\nsmetanin\\PycharmProjects\\nfb\\pynfb\\signals\\_bci_dev\\sm_ksenia_2.mat'
[eeg_data2, states_labels2, _fs, _chan_names, _chan_numb, _samp_numb, _states_codes] = BCIModel.open_eeg_mat(filename, centered=False)
eeg_data = eeg_data1#np.concatenate([DCBlocker().filter(eeg_data1.T).T, DCBlocker().filter(eeg_data2.T).T], 1)

print(eeg_data.shape)
fs = fs[0, 0]
nozeros_mask = np.sum(eeg_data[:, :fs * 2], 1) != 0  # Detect constant (zero) channels
without_emp_mask = nozeros_mask & (chan_names[0, :] != 'A1') & (chan_names[0, :] != 'A2') & (chan_names[0, :] != 'AUX')
eeg_data = eeg_data[without_emp_mask, :].T  # Remove constant (zero) channels and prespecified channels
ch_names = [name[0][0] for name in chan_names[:, without_emp_mask].T]
labels = states_labels1[0]#np.concatenate([states_labels1, states_labels2], 1)[0]
n_samples = len(labels)
n_channels = len(ch_names)
print('eeg_data shape:', eeg_data.shape, 'labels shape:', labels.shape)
#plt.plot(eeg_data + np.arange(n_channels))

#ButterFilter((0.5, 45), fs, eeg_data)
#eeg_data = BCIModel.butter_bandpass_filter(eeg_data, 0.5, 45, fs, order=5, how_to_filt='separately')

# prefilter
# TODO: filtfilt?
#eeg_data = DCBlocker().filter(eeg_data)
#eeg_data = ButterFilter((0.5, 45), fs, n_channels).apply(eeg_data)
#eeg_data, labels = BCIModel.remove_outliers(eeg_data.T, labels[None,:], 7)
#eeg_data = eeg_data.T
#n_samples = eeg_data.shape[0]
#from pynfb.signal_processing.helpers import get_outliers_mask
#eeg_data[get_outliers_mask(eeg_data), :] = 0





bands = [(6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 22), (20, 24)]
state_labels = [1, 2, 6]
csp_multi_class = FilterStack([])
for label in [1, 2, 6]:
    csp_pool = SpatialDecompositionPool(ch_names, fs, bands)
    csp_pool.fit(eeg_data, labels==label)
    csp_multi_class.stack.append(csp_pool.get_filter_stack())
    plt.fill_between(np.arange(n_samples-1000), 0* (labels[1000:] == label), 3 * len(bands)*2 * (labels[1000:] == label), alpha=0.5)





eeg_data = csp_multi_class.apply(eeg_data)[1000:]
labels = labels[1000:]
n_samples = len(eeg_data)

eeg_data = InstantaneousVarianceFilter(eeg_data.shape[1], n_taps=fs//2).apply(eeg_data)
#eeg_data = eeg_data/eeg_data.std(0)

plt.plot(eeg_data/eeg_data.std(0)/5 + np.arange(len(eeg_data[0])))

j = 0
for label in [1, 2, 6]:
    for ch in bands:
        j += 2
        plt.text(0, j, str(label) + str(ch))


from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#clf = OneVsRestClassifier(LogisticRegression(random_state=0, penalty='l1'))
clf = MLPClassifier(hidden_layer_sizes=(), early_stopping=True, verbose=True)

k = 2*n_samples//3
train_slice = slice(None, k)
test_slice = slice(k, None)
acc_train = sum(clf.fit(eeg_data[train_slice], labels[train_slice]).predict(eeg_data[train_slice]) == labels[train_slice])/len(labels[train_slice])
acc_test = sum(clf.predict(eeg_data[test_slice]) == labels[test_slice])/len(labels[test_slice])
print('train', acc_train)
print('test', acc_test)



plt.show()