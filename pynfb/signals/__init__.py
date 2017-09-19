import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import savgol_coeffs, lfiltic, lfilter, butter
from .composite import CompositeSignal
from .bci import BCISignal
from .rejections import Rejections
from pynfb.io import save_spatial_filter
from pynfb.signal_processing.filters import FFTBandEnvelopeDetector, ButterBandEnvelopeDetector, ComplexDemodulationBandEnvelopeDetector,\
    ExponentialSmoother, SGSmoother, ScalarButterFilter, IdentityFilter

ENVELOPE_DETECTOR_TYPE_DEFAULT = 'fft'
ENVELOPE_DETECTOR_KWARGS_DEFAULT = {
    'fft': {
        'n_samples': 500,
        'smoothing_factor': 0.1
    },
    'savgol': {
        'n_samples': 151,
        'order': 2
    },
    'identity': {
        'n_samples': 1000
    }
}


class DerivedSignal():
    def __init__(self, ind, source_freq, n_channels=50, n_samples=1000, bandpass_low=None, bandpass_high=None,
                 spatial_filter=None, scale=False, name='Untitled', disable_spectrum_evaluation=False,
                 smoothing_factor=0.1, temporal_filter_type='fft', envelop_detector_kwargs=None, smoother_type='exp',
                 estimator_type='envdetector', filter_order=2):

        self.n_samples = int(n_samples)

        # bandpass
        self.bandpass = (bandpass_low if bandpass_low else 0,
                         bandpass_high if bandpass_high else source_freq)

        if estimator_type == 'envdetector':
            # setup smoother
            if smoother_type == 'exp':
                smoother = ExponentialSmoother(smoothing_factor)
            elif smoother_type == 'savgol':
                smoother = SGSmoother(151, 2)
            else:
                raise TypeError('Incorrect smoother type')
            # setup specific parameters of envelope detector
            if temporal_filter_type == 'fft':
                self.signal_estimator = FFTBandEnvelopeDetector(self.bandpass, source_freq, smoother, self.n_samples)
            elif temporal_filter_type == 'complexdem':
                self.signal_estimator = ComplexDemodulationBandEnvelopeDetector(self.bandpass, source_freq, smoother)
            elif temporal_filter_type == 'butter':
                self.signal_estimator = ButterBandEnvelopeDetector(self.bandpass, source_freq, smoother, filter_order)
            else:
                raise TypeError('Incorrect envelope detector type')
        elif estimator_type == 'filter':
            self.signal_estimator = ScalarButterFilter(self.bandpass, source_freq, filter_order)
        elif estimator_type == 'identity':
            self.signal_estimator = IdentityFilter()
        else:
            raise TypeError('Incorrect estimator type')

        # id
        self.ind = ind
        # signal name
        self.name = name

        # signal buffer
        self.buffer = np.zeros((self.n_samples,))
        # signal statistics
        self.scaling_flag = scale
        self.mean = np.nan
        self.std = np.nan
        # signal statistics accumulators
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

        # rejections matrices list
        self.rejections = Rejections(n_channels)

        # spatial filter
        self.spatial_filter = np.zeros((n_channels,))
        self.spatial_filter_topography = None
        if spatial_filter is None:
            self.spatial_filter[0] = 0
        else:
            shape = min(spatial_filter.shape[0], n_channels)
            self.spatial_filter[:shape] = spatial_filter[:shape]

        # spatial matrix
        self.spatial_matrix = self.spatial_filter.copy()

        # current sample
        self.current_sample = 0
        self.previous_sample = 0
        self.current_chunk = None
        pass

    def spatial_filter_is_zeros(self):
        return (self.spatial_filter == 0).all()

    def update(self, chunk):

        # spatial filter
        chunk_size = len(chunk)
        filtered_chunk = np.dot(chunk, self.spatial_matrix)
        current_chunk = self.signal_estimator.apply(filtered_chunk)

        # accumulate sum and sum^2
        self.current_sample = filtered_chunk
        self.mean_acc = (self.n_acc * self.mean_acc + current_chunk.sum()) / (self.n_acc + chunk_size)
        self.var_acc = (self.n_acc * self.var_acc + (current_chunk - self.mean_acc).sum() ** 2) / (
                self.n_acc + chunk_size)
        self.std_acc = self.var_acc ** 0.5
        self.n_acc += chunk_size

        if self.scaling_flag and self.std > 0:
            current_chunk = (current_chunk - self.mean) / self.std

        self.current_chunk = current_chunk
        self.current_sample = current_chunk[-1]
        pass

    def update_statistics(self, raw=None, emulate=False, from_acc=False,
                          signals_recorder=None, stats_previous=None, drop_outliers=0):
        if from_acc:
            self.mean = self.mean_acc
            self.std = self.std_acc
            self.reset_statistic_acc()
            return None

        if raw is not None and emulate:
            signal_recordings = np.zeros_like(signals_recorder[:, self.ind])
            self.reset_statistic_acc()
            mean_chunk_size = 8
            for k in range(0, raw.shape[0] - mean_chunk_size, mean_chunk_size):
                chunk = raw[k:k + mean_chunk_size]
                self.update(chunk)
                signal_recordings[k:k + mean_chunk_size] = self.current_chunk
        else:
            signal_recordings = signals_recorder[:, self.ind]
            mean_prev, std_prev = stats_previous[self.ind]
            if np.isfinite(mean_prev) and np.isfinite(std_prev):
                signal_recordings = signal_recordings * std_prev + mean_prev
        # drop outliers:
        if drop_outliers and signal_recordings.std() > 0:
                signal_recordings_clear = signal_recordings[
                    np.abs(signal_recordings - signal_recordings.mean()) < drop_outliers * signal_recordings.std()]
        else:
            signal_recordings_clear = signal_recordings
        self.mean = signal_recordings_clear.mean()
        self.std = signal_recordings_clear.std()
        return (signal_recordings - self.mean) / (self.std if self.std > 0 else 1)

    def update_spatial_filter(self, spatial_filter=None, topography=None):
        if spatial_filter is not None:
            self.spatial_filter = np.array(spatial_filter)
        self.spatial_matrix = np.dot(self.rejections.get_prod(), self.spatial_filter)
        self.spatial_filter_topography = topography if topography is not None else self.spatial_filter_topography

    def update_rejections(self, rejections, append=False):
        self.rejections.update_list(rejections, append=append)
        self.update_spatial_filter()

    def update_ica_rejection(self, rejection=None):
        self.rejections.update_ica(rejection)
        self.update_spatial_filter()

    def update_bandpass(self, bandpass):
        self.bandpass = bandpass

    def drop_rejection(self, ind):
        self.rejections.drop(ind)
        self.update_spatial_filter()
        print(self.rejections)

    def reset_statistic_acc(self):
        self.mean_acc = 0
        self.var_acc = 0
        self.std_acc = 0
        self.n_acc = 0

    def enable_scaling(self):
        self.scaling_flag = True
        pass

    def save_spatial_matrix(self, file_path, channels_labels=None):
        """
        Save full spatial matrix: R1*R2*...*Rk*S, where R1,..Rk - rejections matrices, S - spatial filter
        :return:
        """
        save_spatial_filter(file_path, self.spatial_matrix, channels_labels=channels_labels)

