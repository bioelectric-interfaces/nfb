from pylsl import StreamInlet, resolve_stream

import time
import numpy as np
from scipy import io
from scipy.signal import butter, lfilter
from scipy.linalg import eig
import matplotlib.pyplot as plt
import theano
import os
import scipy.signal as spsig
import pickle
#from bcimodel_draft import BCIModel

#os.chdir('D:/BCI_MINI/recordings')


class BCISignal():
    def __init__(self, model):
        self.model = model

        self.all_CSPs = self.model.all_CSPs
        filtorder = self.model.filt_order
        self.win = self.model.win_av
        self.freq = self.model.freqs
        srate = self.model.srate
        self.M_eog = self.model.M_eog

        fmin_pre = self.model.pre_fh
        fmax_pre = self.model.pre_fl

        self.without_emp_mask = self.model.without_emp_mask

        ##NN

        # ____________________Set NN____________________

        X = theano.tensor.fmatrix('X')
        Y = theano.tensor.imatrix('Y')

        # initialize weights and biases
        W1_init = self.model.W1_saved
        b1_init = self.model.b1_saved
        W2_init = self.model.W2_saved
        b2_init = self.model.b2_saved

        W1 = theano.shared(W1_init, name='W1')
        b1 = theano.shared(b1_init, name='b1')
        W2 = theano.shared(W2_init, name='W2')
        b2 = theano.shared(b2_init, name='b2')

        dot_1 = theano.tensor.dot(X, W1) + b1
        activ_1 = theano.tensor.nnet.sigmoid(dot_1)
        dot_2 = theano.tensor.dot(activ_1, W2) + b2
        activ_final = theano.tensor.nnet.softmax(dot_2)

        pred_y = theano.tensor.argmax(activ_final)

        # Function to get classes' probabilities for a given input
        pred_proba = theano.function([X], activ_final, allow_input_downcast=True)

        # Function to predict class for a given input
        self.predict_val = theano.function([X], pred_y, allow_input_downcast=True)

        # Function to check accuracy on a dataset (proportion of correct)
        def accuracy_for_dataset(inputs, labels):
            return sum([self.predict_val(inputs[i, :].reshape(1, inputs.shape[1])) == labels[i] for i in
                        range(inputs.shape[0])]) / float(inputs.shape[0])

        ######

        numf = len(self.model.freqs)
        numc = 0

        num_all = len(self.all_CSPs)
        for i in range(num_all):
            if (self.all_CSPs[i].shape[0] > 0):
                numc = numc + 1

        numch = 24  # TODO

        # preprocessing high and lowpass filters

        self.a_pre_high = np.zeros((1, filtorder + 1))
        self.b_pre_high = np.zeros((1, filtorder + 1))
        self.a_pre_low = np.zeros((1, filtorder + 1))
        self.b_pre_low = np.zeros((1, filtorder + 1))

        self.Zlast_pre_high = np.zeros((1, numch, filtorder))
        self.Zlast_pre_low = np.zeros((1, numch, filtorder))

        [self.b_pre_high[0], self.a_pre_high[0]] = spsig.butter(filtorder, float(fmin_pre) / (srate / 2), 'high')
        [self.b_pre_low[0], self.a_pre_low[0]] = spsig.butter(filtorder, float(fmax_pre) / (srate / 2), 'low')

        # create bandpass filters

        self.a_high = np.zeros((numf, filtorder + 1))
        self.b_high = np.zeros((numf, filtorder + 1))
        self.a_low = np.zeros((numf, filtorder + 1))
        self.b_low = np.zeros((numf, filtorder + 1))

        numch_without_emp = np.sum(self.without_emp_mask)

        self.Zlast_high = np.zeros((numf, numch_without_emp, filtorder))
        self.Zlast_low = np.zeros((numf, numch_without_emp, filtorder))

        for fr in range(numf):
            fmin = self.freq[fr][0]
            fmax = self.freq[fr][1]

            [self.b_high[fr], self.a_high[fr]] = spsig.butter(filtorder, float(fmin) / (srate / 2), 'high')
            [self.b_low[fr], self.a_low[fr]] = spsig.butter(filtorder, float(fmax) / (srate / 2), 'low')

        # create moving average filter
        self.Zlast_ma = np.zeros((numc, self.win - 1))
        a_ma = 1
        b_ma = np.ones((1, self.win))

        ###




        exptime = 900
        dt_rate = 0.1

        # allocating buffers
        self.received_data_buf = np.zeros((numch, int(exptime * srate * 1.2)))
        self.states_predicted_buf = np.zeros((1, int(exptime * srate * 1.2)))
        self.pos = 0
        self.pos_pred = 0

        # first resolve an EEG stream on the lab network
        #print("looking for an EEG stream...")
        #streams = resolve_stream('type', 'Data')

        # create a new inlet to read from the stream
        #inlet = StreamInlet(streams[0])

        #globalstart = time.time()

    def update(self, chunk):
        np_ar_chunk = np.asarray(chunk)
        chunk_size = np_ar_chunk.shape[0]

        if chunk_size > 0:

            data_chunk_test = np_ar_chunk.T

            self.received_data_buf[:, self.pos:(self.pos + chunk_size)] = data_chunk_test
            self.pos = self.pos + chunk_size + 1

            [data_chunk_test, self.Zlast_pre_high[0, :, :]] = spsig.lfilter(self.b_pre_high[0], self.a_pre_high[0], data_chunk_test, 1,
                                                                       self.Zlast_pre_high[0, :, :])
            [data_chunk_test, self.Zlast_pre_low[0, :, :]] = spsig.lfilter(self.b_pre_low[0], self.a_pre_low[0], data_chunk_test, 1,
                                                                      self.Zlast_pre_low[0, :, :])

            data_chunk_test = data_chunk_test[self.without_emp_mask, :]
            # chan_names_test_used = chan_names_test[:,without_emp_mask]
            data_chunk_test = np.dot(self.M_eog, data_chunk_test)

            ###filt_apply_CSPs(data, sr, freq_range, self.all_CSPs, how_to_filt, self.win, order=5, normalize=False):

            N_csp_per_freq = len(self.all_CSPs) // len(self.freq)
            all_CSPs_copy = list(self.all_CSPs)
            transformed_data_chunk = np.zeros((0, data_chunk_test.shape[1]))

            for fr_ind in range(len(self.freq)):

                [data_chunk_test, self.Zlast_high[fr_ind, :, :]] = spsig.lfilter(self.b_high[fr_ind], self.a_high[fr_ind],
                                                                            data_chunk_test, 1,
                                                                            self.Zlast_high[fr_ind, :, :])
                [data_chunk_test, self.Zlast_low[fr_ind, :, :]] = spsig.lfilter(self.b_low[fr_ind], self.a_low[fr_ind],
                                                                           data_chunk_test, 1, self.Zlast_low[fr_ind, :, :])

                for csp_ind in range(N_csp_per_freq):
                    transformed_data_chunk = np.vstack(
                        (transformed_data_chunk, np.dot(all_CSPs_copy.pop(0), data_chunk_test)))

            final_data_chunk = transformed_data_chunk[1:, :] ** 2
            a_ma = 1
            b_ma = np.ones(self.win) / float(self.win)

            [final_data_chunk, self.Zlast_ma] = spsig.lfilter(b_ma, a_ma, transformed_data_chunk, 1, self.Zlast_ma)

            inputs = final_data_chunk

            chunk_res = np.array(
                [self.predict_val(inputs[:, i].reshape(1, inputs.shape[0])) for i in range(inputs.shape[1])])

            res_chunk_size = chunk_res.shape[0]
            self.states_predicted_buf[:, self.pos_pred:(self.pos_pred + res_chunk_size)] = chunk_res
            self.pos_pred = self.pos_pred + res_chunk_size

            print(chunk_res[-1])

    def fit_model(self, data):
        self.model.fit(data)



if __name__ == '__main__':
    from pynfb.signals._bci_dev.bcimodel_draft import BCIModel
    model = BCIModel()
    model.fit('sm_ksenia_1.mat')
    signal = BCISignal(model=model)
    chunk = np.ones((10, 24))
    data = BCIModel.open_eeg_mat('sm_ksenia_1.mat')[0].T
    for k in range(len(data)):
        signal.update(data[k:(k+1)])
