# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:56:09 2017

@author: voxxys
"""

import time
import numpy as np
from scipy import io
from scipy.signal import butter, lfilter
from scipy.linalg import eig
import matplotlib.pyplot as plt
import theano
from functools import reduce
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression #Lasso


class BCIModel():
    def __init__(self):
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
        
        
    def open_eeg_mat(filename, centered=True):
        all_data = io.loadmat(filename)
        eeg_data = all_data['data_cur']
        if centered:
            eeg_data = eeg_data - np.mean(eeg_data,1)[np.newaxis].T
            print( 'Data were centered: channels are zero-mean')
        states_labels = all_data['states_cur']
        states_codes = list(np.unique(states_labels)[:])
        sampling_rate = all_data['srate']
        chan_names = all_data['chan_names']
        return eeg_data, states_labels, sampling_rate, chan_names, eeg_data.shape[0], eeg_data.shape[1], states_codes

    def butter_bandpass(lowcut, highcut, sampling_rate, order=5):
        nyq_freq = sampling_rate*0.5
        low = lowcut/nyq_freq
        high = highcut/nyq_freq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_high_low_pass(lowcut, highcut, sampling_rate, order=5):
        nyq_freq = sampling_rate*0.5
        lower_bound = lowcut/nyq_freq
        higher_bound = highcut/nyq_freq
        b_high, a_high = butter(order, lower_bound, btype='high')
        b_low, a_low = butter(order, higher_bound, btype='low')
        return b_high, a_high, b_low, a_low
    
    def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5, how_to_filt = 'separately'):
        if how_to_filt == 'separately':
            b_high, a_high, b_low, a_low = BCIModel.butter_high_low_pass(lowcut, highcut, sampling_rate, order=order)
            y = lfilter(b_high, a_high, data)
            y = lfilter(b_low, a_low, y)
        elif how_to_filt == 'simultaneously':
            b, a = BCIModel.butter_bandpass(lowcut, highcut, sampling_rate, order=order)
            y = lfilter(b, a, data)
        return y
    
    def remove_outliers(data_raw, states_labels_raw, iter_numb):
        data = np.copy(data_raw)
        states_labels = np.copy(states_labels_raw)
        data_pwr = np.sqrt(np.sum(data**2,0))
        
        for i in range(iter_numb):
            X_mean = np.mean(data_pwr)
            X_std = np.std(data_pwr)
            mask = np.abs(data_pwr - X_mean) < 2.5*np.abs(X_std)
            data = data[:, mask]
            states_labels = states_labels[:, mask]
            data_pwr = data_pwr[mask]
            print('Samples left after outliers removal:', data_pwr.shape[0])
            
        return data, states_labels
    
    
    def remove_eog_simple(data, chan_names, eyechan, N_art_comp=3):
        
        only_eye_chan = data[chan_names[0,:]==eyechan,:]
        exceed_mask = only_eye_chan > 3*np.mean(np.absolute(only_eye_chan))
        print( 'Number of samples identified as containing eye artifacts:', np.sum(exceed_mask))
        U, S, V = np.linalg.svd(data[:, exceed_mask[0,:]], full_matrices=True)
        M_eog = np.eye(U.shape[0])-np.dot(U[:,0:N_art_comp],U[:,0:N_art_comp].T)
        
        return np.dot(M_eog,data), M_eog

    def outer_n(n):
        return np.array(list(range(n))+list(range(-n,0)))
    
    def whitener(C, rtol=1e-15):
        e, E = np.linalg.eigh(C)
        return reduce(np.dot, [E, np.diag(np.where(e > np.max(e) * rtol, e, np.inf)**-0.5), E.T])
    
    def csp_base(C_a, C_b):
        P = BCIModel.whitener(C_a + C_b)
        P_C_b = reduce(np.dot, [P, C_b, P.T])
        _, _, B = np.linalg.svd((P_C_b))
        return np.dot(B, P.T)
    
    def csp(C_a, C_b, m):
        W = BCIModel.csp_base(C_a, C_b)
        assert W.shape[1] >= 2*m
        return W[BCIModel.outer_n(m)]
        
    def get_CSP_matr(data, states_labels, main_state, N_comp, other_state=None, mode='one_vs_all'):
        
        A = data[:, (states_labels == main_state)[0,:]]
        if mode == 'one_vs_all':
            B = data[:, (states_labels != main_state)[0,:]]
        elif mode == 'pairwise':
            if other_state == None:
                print( "Other state must be specified")
                return None
            else:
                B = data[:, (states_labels == other_state)[0,:]]
                
        C1 = np.cov(A)
        C2 = np.cov(B)
        
        return BCIModel.csp(C1,C2,N_comp)

    def const_features(data,states_labels,states_codes,sr,feat_type,freq_ranges,how_to_filt,N_csp_comp,win,order=5,normalize=False):
        '''
        Filters data according to specified bands (in freq_range) and derives CSP transformations for each band.
        Type of CSP should be provided in feat_type: pairwise or one-vs-all (recommended).
        Number of CSP components should be provided in N_csp_comp (for a given N, N first and N last components will be used).
        Time interval for averaging is specified in win.
        If normalize=True, each data point is in [0,1].
        Returns array of transformed data, list of CSP transform matrices (in arrays), 
        and array of state codes for each of the final features (i.e., which state was first while this CSP projection was computed)
        '''
        final_data = np.zeros((1, data.shape[1]))
        all_CSPs = []
        if feat_type == 'CSP_pairwise':
            where_states = []
            for freq in freq_ranges:
                data_filt = BCIModel.butter_bandpass_filter(data, freq[0], freq[1], sr, order, how_to_filt)
                all_states_CSP = []
                for st in states_codes:
                    for oth_st in np.array(states_codes)[np.array(states_codes)!=st]:
                        CSP_st = BCIModel.get_CSP_matr(data_filt, states_labels, st, N_csp_comp, other_state=oth_st, mode='pairwise')
                        all_states_CSP.append(np.dot(CSP_st, data_filt))
                        all_CSPs.append(CSP_st)
                        where_states.extend([st]*(N_csp_comp*2))
                    data_transformed = np.vstack(all_states_CSP)**2
                final_data = np.vstack((final_data, data_transformed))
                
        elif feat_type == 'CSP_one_vs_all':
            where_states = []
            for freq in freq_ranges:
                data_filt = BCIModel.butter_bandpass_filter(data, freq[0], freq[1], sr, order, how_to_filt)
                all_states_CSP = []
                for st in states_codes:
                    CSP_st = BCIModel.get_CSP_matr(data_filt, states_labels, st, N_csp_comp, other_state=None, mode='one_vs_all')
                    all_states_CSP.append(np.dot(CSP_st, data_filt))
                    all_CSPs.append(CSP_st)
                    where_states.extend([st]*(N_csp_comp*2))
                data_transformed = np.vstack(all_states_CSP)**2
                final_data = np.vstack((final_data, data_transformed))
    
        final_data = final_data[1:,:]
        a_ma = 1
        b_ma = np.ones(int(win))/float(win)
        final_data = lfilter(b_ma, a_ma, final_data)
        if normalize:
            final_data = final_data/np.sum(final_data,0)[np.newaxis,:]
        print( 'Shape of data matrix:', final_data.shape)
            
        return final_data, all_CSPs, np.array(where_states)
    
    def filt_apply_CSPs(data, sr, freq_range, all_CSPs, how_to_filt, win, order=5, normalize=False):
        '''
        Filters data according to specified bands (in freq_range) and applies corresponding CSP transformations (in all_CSPs).
        Order in freq_range and all_CSPs must be the same.
        If normalize=True, each data point is in [0,1].
        '''
        N_csp_per_freq = len(all_CSPs)/len(freq_range)
        all_CSPs_copy = list(all_CSPs)
        transformed_data = np.zeros((1, data.shape[1]))
        for fr_ind in range(len(freq_range)):
            filt_data = BCIModel.butter_bandpass_filter(data,freq_range[fr_ind][0],freq_range[fr_ind][1],sr,order,how_to_filt)
            for csp_ind in range(N_csp_per_freq):
                transformed_data = np.vstack((transformed_data, np.dot(all_CSPs_copy.pop(0), filt_data)))
        final_data = transformed_data[1:,:]**2
        a_ma = 1
        b_ma = np.ones(win)/float(win)
        final_data = lfilter(b_ma, a_ma, final_data)
        if normalize:
            final_data = final_data/np.sum(final_data,0)[np.newaxis,:]
        return final_data    
            
    def lasso(eeg_data, states_labels, states_codes, where_states, frequences, N_CSP_comp, all_CSPs, tuned_parameters):

        
        clf_l1_LR_1vsAll = GridSearchCV(LogisticRegression(penalty='l1'), tuned_parameters, cv=3)
        clf_l1_LR_2vsAll = GridSearchCV(LogisticRegression(penalty='l1'), tuned_parameters, cv=3)
        clf_l1_LR_6vsAll = GridSearchCV(LogisticRegression(penalty='l1'), tuned_parameters, cv=3)
        #clf_l1_LR_1vsAll = LogisticRegression(C=0.001, penalty='l1')#Lasso(alpha=0.01)
        #clf_l1_LR_2vsAll = LogisticRegression(C=0.001, penalty='l1')#Lasso(alpha=0.01)
        #clf_l1_LR_6vsAll = LogisticRegression(C=0.001, penalty='l1')#Lasso(alpha=0.01)
        
        data_state_1 = eeg_data[(where_states==1), :]
        data_state_1 = data_state_1[:, (states_labels == 1)[0,:]]
        data_states_NOT_1 = eeg_data[(where_states==1), :]
        data_states_NOT_1 = data_states_NOT_1[:, (states_labels != 1)[0,:]]
        clf_l1_LR_1vsAll.fit(np.hstack((data_state_1, data_states_NOT_1)).T, 
                             np.vstack((np.ones((data_state_1.shape[1],1)),np.zeros((data_states_NOT_1.shape[1],1)))).ravel())
        
        data_state_2 = eeg_data[(where_states==2), :]
        data_state_2 = data_state_2[:, (states_labels == 2)[0,:]]
        data_states_NOT_2 = eeg_data[(where_states==2), :]
        data_states_NOT_2 = data_states_NOT_2[:, (states_labels != 2)[0,:]]
        clf_l1_LR_2vsAll.fit(np.hstack((data_state_2, data_states_NOT_2)).T, 
                            np.vstack((np.ones((data_state_2.shape[1],1)),np.zeros((data_states_NOT_2.shape[1],1)))).ravel())
        
        data_state_6 = eeg_data[(where_states==6), :]
        data_state_6 = data_state_6[:, (states_labels == 6)[0,:]]
        data_states_NOT_6 = eeg_data[(where_states==6), :]
        data_states_NOT_6 = data_states_NOT_6[:, (states_labels != 6)[0,:]]
        clf_l1_LR_6vsAll.fit(np.hstack((data_state_6, data_states_NOT_6)).T, 
                            np.vstack((np.ones((data_state_6.shape[1],1)),np.zeros((data_states_NOT_6.shape[1],1)))).ravel())
        
        all_coef_lasso = [clf_l1_LR_1vsAll.best_estimator_.coef_, clf_l1_LR_2vsAll.best_estimator_.coef_, 
                          clf_l1_LR_6vsAll.best_estimator_.coef_]
        for ind_coef_matr in range(len(all_coef_lasso)):
            dummy_matr = all_coef_lasso[ind_coef_matr].reshape((len(frequences),N_CSP_comp*2))
            print( '\n',states_codes[ind_coef_matr],'vs all:')
            for i in range(len(frequences)):
                if np.count_nonzero(dummy_matr[i,:])!=0:
                    print( 'Frequency band', frequences[i], 'provided CSP components:', np.nonzero(dummy_matr[i,:])[0]+1)
                    
        
        # Remove irrelevant (according to lasso LR) features 
        masks_1 = np.split((clf_l1_LR_1vsAll.best_estimator_.coef_!=0), len(frequences), axis=1)
        masks_2 = np.split((clf_l1_LR_2vsAll.best_estimator_.coef_!=0), len(frequences), axis=1)
        masks_6 = np.split((clf_l1_LR_6vsAll.best_estimator_.coef_!=0), len(frequences), axis=1)
        masks_all_ordered = []
        for i in range(len(frequences)):
            masks_all_ordered.extend(masks_1[i])
            masks_all_ordered.extend(masks_2[i])
            masks_all_ordered.extend(masks_6[i])
            
        for j in range(len(masks_all_ordered)):
            all_CSPs[j] = all_CSPs[j][masks_all_ordered[j],:]
        
        mask_data = np.hstack(masks_all_ordered)
        
        eeg_data = eeg_data[mask_data,:]
        print( '\nIrrelevant features have been removed')
        
        return eeg_data, all_CSPs
        
    
    def accuracy_for_dataset(inputs, labels):
        return sum([BCIModel.predict_val(inputs[i,:].reshape(1,inputs.shape[1])) == labels[i] for i in range(inputs.shape[0])])/float(inputs.shape[0])

    def fit(self, filename):
        
        [eeg_data, states_labels, sampling_rate, chan_names, chan_numb, samp_numb, states_codes] = BCIModel.open_eeg_mat(filename, centered=False)
        sampling_rate = sampling_rate[0,0]
        eeg_data = BCIModel.butter_bandpass_filter(eeg_data, 0.5, 45, sampling_rate, order=5, how_to_filt = 'separately')

        nozeros_mask = np.sum(eeg_data[:,:sampling_rate*2],1)!=0 # Detect constant (zero) channels
        without_emp_mask = nozeros_mask & (chan_names[0,:]!='A1') & (chan_names[0,:]!='A2') & (chan_names[0,:]!='AUX')
        eeg_data = eeg_data[without_emp_mask,:] # Remove constant (zero) channels and prespecified channels
        chan_names_used = chan_names[:,without_emp_mask]
        
        # Remove outliers remove artifacts (blinks, eye movements)
        eeg_data, states_labels = BCIModel.remove_outliers(eeg_data, states_labels, 7)
        eeg_data, M_eog = BCIModel.remove_eog_simple(eeg_data,chan_names_used,'Fp1')

        # Construct features: project eeg data on CSP components (separately for each of specified frequency bands)
        N_CSP_comp = 1 # N first and N last 2*N in total
        win = sampling_rate//2 # Window for averaging: 0.5 sec
        frequences = [(6,10),(8,12),(10,14),(12,16),(14,18),(16,20),(18,22),(20,24)]
        eeg_data, all_CSPs, where_states = BCIModel.const_features(eeg_data,states_labels,states_codes,sampling_rate,'CSP_one_vs_all',frequences,'separately',N_CSP_comp,win,normalize=False)
    
        eeg_data = eeg_data[:, win*2:]
        states_labels = states_labels[:, win*2:]
        
        tuned_parameters = [{'C': [0.0005]}]
        
        [eeg_data, all_CSPs] = BCIModel.lasso(eeg_data, states_labels, states_codes, where_states, frequences, N_CSP_comp, all_CSPs, tuned_parameters)
        
        # # #
        
        examples_numb = eeg_data.shape[1]
        train_labels = np.copy(states_labels)
        train_labels[train_labels==1] = 0
        train_labels[train_labels==2] = 1
        train_labels[train_labels==6] = 2
        
        pre_labels_mask = np.zeros((3, examples_numb))
        for row in range(3):
             pre_labels_mask[row, :] = row
        one_of_K_labeled = np.zeros((3, examples_numb))
        for row in range(3):
            one_of_K_labeled[row, :] = (pre_labels_mask[row, :] == train_labels)
        
        # Examples in rows, features in columns
        train_data = eeg_data.T
        train_labels = train_labels.T
        one_of_K_labeled = one_of_K_labeled.T
        print( (train_data.shape,train_labels.shape,one_of_K_labeled.shape))
        
        input_size = train_data.shape[1]
        hidd_size = 15 # Number of units in hidden layer
        out_size = one_of_K_labeled.shape[1]
        print( 'Input dim:', int(input_size), '\nHidden dim:', hidd_size, '\nOutput dim:', int(out_size))

        # # # 

        X = theano.tensor.fmatrix('X')
        Y = theano.tensor.imatrix('Y')
        
        # Randomly initialize weights and biases
        W1_init = np.random.random((input_size,hidd_size)).astype('float32')*0.1
        b1_init = np.random.random((hidd_size)).astype('float32')*0.1
        W2_init = np.random.random((hidd_size,out_size)).astype('float32')*0.1
        b2_init = np.random.random((out_size)).astype('float32')*0.1
        
        W1 = theano.shared(W1_init, name = 'W1')
        b1 = theano.shared(b1_init, name = 'b1')
        W2 = theano.shared(W2_init, name = 'W2')
        b2 = theano.shared(b2_init, name = 'b2')
        
        dot_1 = theano.tensor.dot(X, W1) + b1
        activ_1 = theano.tensor.nnet.sigmoid(dot_1)
        dot_2 = theano.tensor.dot(activ_1, W2) + b2
        activ_final = theano.tensor.nnet.softmax(dot_2)
        
        # Get the index of an output with the max activation (probability)
        pred_y = theano.tensor.argmax(activ_final)
        
        # Define loss function
        cross_loss = theano.tensor.nnet.categorical_crossentropy(activ_final, Y).mean() # Average cross-entropy
        
        # Automatically find expressions for gradients
        g_W2 = theano.tensor.grad(cross_loss, W2)
        g_b2 = theano.tensor.grad(cross_loss, b2)
        g_W1 = theano.tensor.grad(cross_loss, W1)
        g_b1 = theano.tensor.grad(cross_loss, b1)
        
        # Set learning rates for weights and biases
        lr_W1 = np.array(0.09).astype('float32')
        lr_b1 = np.array(0.09).astype('float32')
        lr_W2 = np.array(0.01).astype('float32')
        lr_b2 = np.array(0.01).astype('float32')
        
        # Define how to update weights and biases
        updates_for_params = [(W2, W2 - lr_W2*g_W2),
                              (b2, b2 - lr_b2*g_b2),
                              (W1, W1 - lr_W1*g_W1),
                              (b1, b1 - lr_b1*g_b1)]
        
        # Define theano function that trains network
        train_net = theano.function(inputs = [X, Y],
                                   outputs = cross_loss,
                                   updates = updates_for_params,
                                   allow_input_downcast = True)
        
        # Function to get classes' probabilities for a given input
        BCIModel.pred_proba = theano.function([X], activ_final, allow_input_downcast = True)
        
        # Function to predict class for a given input
        BCIModel.predict_val = theano.function([X], pred_y, allow_input_downcast = True)
        
    
        batch_size = 10
        iter_numb = train_data.shape[0]//batch_size
        max_epoch = 12
        all_mean_loss = []
        
        for epoch_ in range(max_epoch):
            start_time = time.time()
            
            loss_list = []
            for iter_ in np.random.choice(np.arange(0,examples_numb), size=(iter_numb), replace=False):
                loss_list.append(train_net(train_data[iter_:iter_+batch_size,:], one_of_K_labeled[iter_:iter_+batch_size,:]))
            
            mean_loss = np.mean(loss_list)
            
            print( 'Time passed (min):', (time.time()-start_time)/float(60))
            #if not epoch_ % 10:
            print( 'Epoch '+str(epoch_)+':\naverage loss is '+str(mean_loss)+'\n')
        
            all_mean_loss.append(mean_loss)
            if epoch_>0 and (all_mean_loss[-2]-all_mean_loss[-1])>0.0 and (all_mean_loss[-2]-all_mean_loss[-1])<0.0001:
                break

                
        # Check accuracy on training dataset
        print( 'Train accuracy:', BCIModel.accuracy_for_dataset(train_data, train_labels))
                
        self.all_CSPs = all_CSPs
        self.freqs = frequences
        self.win_av = win
        self.srate = sampling_rate
        self.filt_order = 5
        self.pre_fl = 45
        self.pre_fh = 0.5
        self.M_eog = M_eog
        self.without_emp_mask = without_emp_mask
        self.W1_saved = W1.get_value()
        self.b1_saved = b1.get_value()
        self.W2_saved = W2.get_value()
        self.b2_saved = b2.get_value()


        pass

class BCISignal():
    def __init__(self, arg1):
        self.model = BCIModel(n_samples=42)
        self.current_state = 0
        
        X = theano.tensor.fmatrix('X')
        
        # Randomly initialize weights and biases
        W1_init = self.model.W1
        b1_init = self.model.b2
        W2_init = self.model.W2
        b2_init = self.model.b2
        
        W1 = theano.shared(W1_init, name = 'W1')
        b1 = theano.shared(b1_init, name = 'b1')
        W2 = theano.shared(W2_init, name = 'W2')
        b2 = theano.shared(b2_init, name = 'b2')
        
        dot_1 = theano.tensor.dot(X, W1) + b1
        activ_1 = theano.tensor.nnet.sigmoid(dot_1)
        dot_2 = theano.tensor.dot(activ_1, W2) + b2
        activ_final = theano.tensor.nnet.softmax(dot_2)
        
        # Get the index of an output with the max activation (probability)
        # pred_y = theano.tensor.argmax(activ_final)
        
        pred_y = theano.tensor.argmax(activ_final)
        
        # Function to get classes' probabilities for a given input
        self.pred_proba = theano.function([X], activ_final, allow_input_downcast = True)
        
        # Function to predict class for a given input
        self.predict_val = theano.function([X], pred_y, allow_input_downcast = True)

    def update(self, chunk):
        self.current_state = self.model.n_samples**2

    def fit_model(self, data):
        self.model.fit(data)