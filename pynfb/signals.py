import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq





class DerivedSignal():
    def __init__(self, n_channels=50, n_samples=1000, bandpass_low=None, bandpass_high=None, spatial_matrix=None,
                 source_freq=500, scale=False):
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
        if spatial_matrix is None:
            self.spatial_matrix = np.zeros((n_channels, ))
            self.spatial_matrix[0] = 1
        else:
            self.spatial_matrix = spatial_matrix
        # current sample
        self.current_sample = 0

        # bandpass filter settings
        self.w = fftfreq(n_samples, d=1. / source_freq * 2)
        self.bandpass = (bandpass_low if bandpass_low else self.w[0],
                         bandpass_high if bandpass_high else self.w[-1])

        # asymmetric gaussian window
        p  = round(n_samples*2/4) # maximum
        eps = 0.0001 # bounds value
        power = 2 # power of x
        left_c = - np.log(eps)/ (p**power)
        right_c = - np.log(eps) / (n_samples-1-p) ** power
        samples_window= np.concatenate([np.exp(-left_c * abs(np.arange(p) - p) ** power),
                                        np.exp(-right_c * abs(np.arange(p, n_samples) - p) ** power)])
        self.samples_window = samples_window
        pass

    def update(self, chunk):
        # spatial filter
        filtered_chunk = np.dot(chunk, self.spatial_matrix)
        # update buffer
        chunk_size = filtered_chunk.shape[0]
        self.buffer[:-chunk_size] = self.buffer[chunk_size:]
        self.buffer[-chunk_size:] = filtered_chunk
        # bandpass filter and amplitude
        self.current_sample = self.get_bandpass_amplitude()
        if self.scaling_flag:
            self.current_sample = (self.current_sample - self.mean)/self.std
        # accumulate sum and sum^2
        self.mean_acc = (self.n_acc*self.mean_acc + chunk_size*self.current_sample)/(self.n_acc+chunk_size)
        self.var_acc = (self.n_acc*self.var_acc + chunk_size*(self.current_sample - self.mean_acc)**2)/(self.n_acc+chunk_size)
        self.std_acc = self.var_acc**0.5
        self.n_acc += chunk_size
        pass

    def get_bandpass_amplitude(self):
        f_signal = rfft(np.hstack((self.buffer[self.buffer.shape[0]//2+self.buffer.shape[0]%2:], -self.buffer[-1:self.buffer.shape[0]//2+self.buffer.shape[0]%2-1:-1])) * self.samples_window)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(self.w < self.bandpass[0]) | (self.w > self.bandpass[1])] = 0  # TODO: in one row
        amplitude = sum(np.abs(cut_f_signal))
        return amplitude