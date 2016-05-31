import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq





class DerivedSignal():
    def __init__(self, n_channels=50, n_samples=1000, bandpass_low=None, bandpass_high=None, spatial_matrix=None,
                 source_freq=500, scale=False, name='Untitled', disable_spectrum_evaluation=False):
        # signal name
        self.name = name
        # signal buffer
        self.buffer = np.zeros((n_samples, ))
        # signal statistics
        self.scaling_flag = scale
        self.mean = np.nan
        self.std = np.nan
        # signal statistics accumulators
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0
        # spatial matrix
        self.spatial_matrix = np.zeros((n_channels,))
        if spatial_matrix is None:
            self.spatial_matrix[0] = 1
        else:
            shape = min(spatial_matrix.shape[0], n_channels)
            self.spatial_matrix[:shape] = spatial_matrix[:shape]
        # current sample
        self.current_sample = 0
        # bandpass and exponential smoothing flsg
        self.disable_spectrum_evaluation = disable_spectrum_evaluation
        # bandpass filter settings
        self.w = fftfreq(2*n_samples, d=1. / source_freq * 2)
        self.bandpass = (bandpass_low if bandpass_low else 0,
                         bandpass_high if bandpass_high else source_freq)

        # asymmetric gaussian window
        p  = round(2*n_samples*2/4) # maximum
        eps = 0.0001 # bounds value
        power = 2 # power of x
        left_c = - np.log(eps)/ (p**power)
        right_c = - np.log(eps) / (2*n_samples-1-p) ** power
        samples_window= np.concatenate([np.exp(-left_c * abs(np.arange(p) - p) ** power),
                                        np.exp(-right_c * abs(np.arange(p, 2*n_samples) - p) ** power)])
        self.samples_window = samples_window
        pass

    def update(self, chunk):

        # spatial filter
        filtered_chunk = np.dot(chunk, self.spatial_matrix)

        # update buffer
        chunk_size = filtered_chunk.shape[0]
        self.buffer[:-chunk_size] = self.buffer[chunk_size:]
        self.buffer[-chunk_size:] = filtered_chunk

        if not self.disable_spectrum_evaluation:
            # bandpass filter and amplitude
            current_sample = self.get_bandpass_amplitude()
            if self.scaling_flag:
                current_sample = (current_sample - self.mean) / self.std
            # exponential smoothing
            if self.n_acc > 10:
                self.current_sample = 0.1*current_sample+0.9*self.current_sample
            else:
                self.current_sample = current_sample
            # accumulate sum and sum^2
            self.mean_acc = (self.n_acc * self.mean_acc + chunk_size * self.current_sample) / (self.n_acc + chunk_size)
            self.var_acc = (self.n_acc * self.var_acc + chunk_size * (self.current_sample - self.mean_acc) ** 2) / (
            self.n_acc + chunk_size)
        else:
            # accumulate sum and sum^2
            self.current_sample = filtered_chunk
            self.mean_acc = (self.n_acc * self.mean_acc + chunk_size * self.current_sample.sum()) / (self.n_acc + chunk_size)
            self.var_acc = (self.n_acc * self.var_acc + chunk_size * (self.current_sample - self.mean_acc).sum() ** 2) / (
                self.n_acc + chunk_size)

        self.std_acc = self.var_acc**0.5
        self.n_acc += chunk_size
        pass

    def get_bandpass_amplitude(self):
        f_signal = rfft(np.hstack((self.buffer, self.buffer[-1::-1])) * self.samples_window)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(self.w < self.bandpass[0]) | (self.w > self.bandpass[1])] = 0  # TODO: in one row
        amplitude = sum(np.abs(cut_f_signal))
        return amplitude

    def update_statistics(self):
        self.mean = self.mean_acc
        self.std = self.std_acc
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

    def enable_scaling(self):
        self.scaling_flag = True
        pass