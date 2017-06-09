# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:30:53 2015

@author: voxxys
"""
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

os.chdir('D:/BCI_MINI/recordings')

class params:
    def __init__(self):
        self.numch = 0 #TOFIX to fix elsewhere        
        
        self.srate = 0
        self.win_av = 0
        
        self.without_emp_mask = 0
        
        self.pre_fl = 0
        self.pre_fh = 0
        
        self.filt_order = 0
        
        self.M_eog = 0
        self.all_CSPs = 0
        self.freqs = 0
        
        self.W1_saved = 0
        self.b1_saved = 0
        self.W2_saved = 0
        self.b2_saved = 0
        
        
with open('params_0.pkl', 'rb') as input:
        params_1 = pickle.load(input)


all_CSPs = params_1.all_CSPs
filtorder = params_1.filt_order
win = params_1.win_av
freq = params_1.freqs
srate = params_1.srate
M_eog = params_1.M_eog

fmin_pre = params_1.pre_fh
fmax_pre = params_1.pre_fl

without_emp_mask = params_1.without_emp_mask


##NN

# ____________________Set NN____________________

X = theano.tensor.fmatrix('X')
Y = theano.tensor.imatrix('Y')


# initialize weights and biases
W1_init = params_1.W1_saved
b1_init = params_1.b1_saved
W2_init = params_1.W2_saved
b2_init = params_1.b2_saved

W1 = theano.shared(W1_init, name = 'W1')
b1 = theano.shared(b1_init, name = 'b1')
W2 = theano.shared(W2_init, name = 'W2')
b2 = theano.shared(b2_init, name = 'b2')

dot_1 = theano.tensor.dot(X, W1) + b1
activ_1 = theano.tensor.nnet.sigmoid(dot_1)
dot_2 = theano.tensor.dot(activ_1, W2) + b2
activ_final = theano.tensor.nnet.softmax(dot_2)

pred_y = theano.tensor.argmax(activ_final)

# Function to get classes' probabilities for a given input
pred_proba = theano.function([X], activ_final, allow_input_downcast = True)

# Function to predict class for a given input
predict_val = theano.function([X], pred_y, allow_input_downcast = True)

# Function to check accuracy on a dataset (proportion of correct)
def accuracy_for_dataset(inputs, labels):
    return sum([predict_val(inputs[i,:].reshape(1,inputs.shape[1])) == labels[i] for i in range(inputs.shape[0])])/float(inputs.shape[0])


######

numf = len(params_1.freqs)
numc = 0

num_all = len(all_CSPs)
for i in range(num_all):
    if(all_CSPs[i].shape[0] > 0):
        numc = numc + 1

numch = 24 #TOFIX

# preprocessing high and lowpass filters

a_pre_high = np.zeros((1,filtorder + 1))
b_pre_high = np.zeros((1,filtorder + 1))
a_pre_low = np.zeros((1,filtorder + 1))
b_pre_low = np.zeros((1,filtorder + 1))

Zlast_pre_high = np.zeros((1,numch,filtorder))
Zlast_pre_low = np.zeros((1,numch,filtorder))

[b_pre_high[0], a_pre_high[0]] = spsig.butter(filtorder, float(fmin_pre)/(srate/2), 'high')
[b_pre_low[0], a_pre_low[0]] = spsig.butter(filtorder, float(fmax_pre)/(srate/2), 'low')


# create bandpass filters

a_high = np.zeros((numf,filtorder + 1))
b_high = np.zeros((numf,filtorder + 1))
a_low = np.zeros((numf,filtorder + 1))
b_low = np.zeros((numf,filtorder + 1))

numch_without_emp = np.sum(without_emp_mask)

Zlast_high = np.zeros((numf,numch_without_emp,filtorder))
Zlast_low = np.zeros((numf,numch_without_emp,filtorder))

for fr in range(numf):
    
    fmin = freq[fr][0]
    fmax = freq[fr][1]
   
    [b_high[fr], a_high[fr]] = spsig.butter(filtorder, float(fmin)/(srate/2), 'high')
    [b_low[fr], a_low[fr]] = spsig.butter(filtorder, float(fmax)/(srate/2), 'low')

# create moving average filter
Zlast_ma = np.zeros((numc,win-1))
a_ma = 1
b_ma = np.ones((1,win))




###




exptime = 900
dt_rate = 0.1

# allocating buffers
received_data_buf = np.zeros((numch, exptime*srate*1.2))
states_predicted_buf = np.zeros((1, exptime*srate*1.2))
pos = 0
pos_pred = 0


# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'Data')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

globalstart = time.time()

while (time.time() - globalstart < exptime):
    
    startwhile = time.time()

    chunk, timestamp = inlet.pull_chunk()
    np_ar_chunk = np.asarray(chunk)
    chunk_size = np_ar_chunk.shape[0]
    
    if chunk_size > 0:
        
        data_chunk_test = np_ar_chunk.T
        
        received_data_buf[:,pos:(pos+chunk_size)] = data_chunk_test
        pos = pos + chunk_size + 1
        
        [data_chunk_test,Zlast_pre_high[0,:,:]] = spsig.lfilter(b_pre_high[0], a_pre_high[0], data_chunk_test, 1, Zlast_pre_high[0,:,:])
        [data_chunk_test,Zlast_pre_low[0,:,:]] = spsig.lfilter(b_pre_low[0], a_pre_low[0], data_chunk_test, 1, Zlast_pre_low[0,:,:])
        
        data_chunk_test = data_chunk_test[without_emp_mask,:]
        #chan_names_test_used = chan_names_test[:,without_emp_mask]
        data_chunk_test = np.dot(M_eog,data_chunk_test)
        
        ###filt_apply_CSPs(data, sr, freq_range, all_CSPs, how_to_filt, win, order=5, normalize=False):
        
        N_csp_per_freq = len(all_CSPs)/len(freq)
        all_CSPs_copy = list(all_CSPs)
        transformed_data_chunk = np.zeros((0, data_chunk_test.shape[1]))
        
        for fr_ind in range(len(freq)):
            
            [data_chunk_test,Zlast_high[fr_ind,:,:]] = spsig.lfilter(b_high[fr_ind], a_high[fr_ind], data_chunk_test, 1, Zlast_high[fr_ind,:,:])
            [data_chunk_test,Zlast_low[fr_ind,:,:]] = spsig.lfilter(b_low[fr_ind], a_low[fr_ind], data_chunk_test, 1, Zlast_low[fr_ind,:,:])  
             
            for csp_ind in range(N_csp_per_freq):
                transformed_data_chunk = np.vstack((transformed_data_chunk, np.dot(all_CSPs_copy.pop(0), data_chunk_test)))
                
                
        final_data_chunk = transformed_data_chunk[1:,:]**2
        a_ma = 1
        b_ma = np.ones(win)/float(win)
        
        [final_data_chunk,Zlast_ma] = spsig.lfilter(b_ma, a_ma, transformed_data_chunk, 1, Zlast_ma)
        
        inputs = final_data_chunk
        
        chunk_res = np.array([predict_val(inputs[:,i].reshape(1,inputs.shape[0])) for i in range(inputs.shape[1])])
        
        
        res_chunk_size = chunk_res.shape[0]
        states_predicted_buf[:,pos_pred:(pos_pred+res_chunk_size)] = chunk_res
        pos_pred = pos_pred+res_chunk_size

        print(chunk_res[-1])
        
    dif = dt_rate - time.time() + startwhile
    if(dif > 0):
        time.sleep(dif)