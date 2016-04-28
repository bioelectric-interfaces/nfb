import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq


class DerivedSignal():
    def __init__(self, n_channels=50, n_samples=500, bandpass_low=None, bandpass_high=None, spatial_matrix=None,
                 source_freq=500, scale=True):
        # signal buffer
        self.buffer = np.zeros((n_samples, ))
        # signal statistics
        self.scale = scale
        self.mean = 0
        self.std = 1
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
        pass

    def update(self, chunk):
        # spatial filter
        filtered_chunk = np.dot(chunk, self.spatial_matrix)
        # update buffer
        chunk_size = filtered_chunk.shape[0]
        self.buffer[:-chunk_size] = self.buffer[chunk_size:]
        self.buffer[-chunk_size:] =  filtered_chunk
        # bandpass filter and amplitude
        self.current_sample = self.get_bandpass_amplitude()
        if self.scale:
            self.current_sample = (self.current_sample - self.mean)/self.std
        pass

    def get_bandpass_amplitude(self):
        f_signal = rfft(self.buffer)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(self.w < self.bandpass[0]) | (self.w > self.bandpass[1])] = 0  # TODO: in one row
        amplitude = sum(np.abs(cut_f_signal))
        return amplitude