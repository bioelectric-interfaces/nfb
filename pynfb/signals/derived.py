import numpy as np
from pynfb.serializers import save_spatial_filter, read_spatial_filter
from pynfb.signal_processing.filters import ExponentialSmoother, SGSmoother, FFTBandEnvelopeDetector, \
    ComplexDemodulationBandEnvelopeDetector, ButterBandEnvelopeDetector, ScalarButterFilter, IdentityFilter, \
    FilterSequence, DelayFilter, CFIRBandEnvelopeDetector
from pynfb.signals.rejections import Rejections

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


class DerivedSignal:
    @classmethod
    def from_params(cls, ind, fs, n_channels, channels, params, spatial_filter=None):
        if spatial_filter is None:
            spatial_filter = read_spatial_filter(params['SpatialFilterMatrix'], fs, channels, params['sROILabel'])
        return cls(ind=ind,
                   bandpass_high=params['fBandpassHighHz'],
                   bandpass_low=params['fBandpassLowHz'],
                   name=params['sSignalName'],
                   n_channels=n_channels,
                   spatial_filter=spatial_filter,
                   disable_spectrum_evaluation=params['bDisableSpectrumEvaluation'],
                   n_samples=params['fFFTWindowSize'],
                   smoothing_factor=params['fSmoothingFactor'],
                   source_freq=fs,
                   estimator_type=params['sTemporalType'],
                   temporal_filter_type=params['sTemporalFilterType'],
                   smoother_type=params['sTemporalSmootherType'],
                   filter_order=params['fTemporalFilterButterOrder'],
                   delay_ms=params['iDelayMs'])

    def __init__(self, ind, source_freq, n_channels=50, n_samples=1000, bandpass_low=None, bandpass_high=None,
                 spatial_filter=None, scale=False, name='Untitled', disable_spectrum_evaluation=False,
                 smoothing_factor=0.1, temporal_filter_type='fft', envelop_detector_kwargs=None, smoother_type='exp',
                 estimator_type='envdetector', filter_order=2, delay_ms=0):

        self.n_samples = int(n_samples)
        self.fs = source_freq
        self.delay_ms = delay_ms

        self.estimator_type = estimator_type
        self.smoother_type = smoother_type
        self.smoothing_factor = smoothing_factor
        self.temporal_filter_type = temporal_filter_type
        self.filter_order = filter_order

        # bandpass
        self.bandpass = (bandpass_low if bandpass_low else 0,
                         bandpass_high if bandpass_high else source_freq)

        self.signal_estimator = self.reset_signal_estimator()

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
        self.previous_sample = 0
        self.current_chunk = None
        pass

    def reset_signal_estimator(self):
        if self.estimator_type == 'envdetector':
            # setup smoother
            if self.smoother_type == 'exp':
                smoother = ExponentialSmoother(self.smoothing_factor)
            elif self.smoother_type == 'savgol':
                smoother = SGSmoother(151, 2)
            else:
                raise TypeError('Incorrect smoother type')
            # setup specific parameters of envelope detector
            if self.temporal_filter_type == 'fft':
                self.signal_estimator = FFTBandEnvelopeDetector(self.bandpass, self.fs, smoother, self.n_samples)
            elif self.temporal_filter_type == 'complexdem':
                self.signal_estimator = ComplexDemodulationBandEnvelopeDetector(self.bandpass, self.fs, smoother)
            elif self.temporal_filter_type == 'butter':
                self.signal_estimator = ButterBandEnvelopeDetector(self.bandpass, self.fs, smoother, self.filter_order)
            elif self.temporal_filter_type == 'cfir':
                self.signal_estimator = CFIRBandEnvelopeDetector(self.bandpass, self.fs, smoother, n_taps=self.n_samples)
            else:
                raise TypeError('Incorrect envelope detector type')
        elif self.estimator_type == 'filter':
            self.signal_estimator = ScalarButterFilter(self.bandpass, self.fs, self.filter_order)
        elif self.estimator_type == 'identity':
            self.signal_estimator = IdentityFilter()
        else:
            raise TypeError('Incorrect estimator type')

        if self.delay_ms > 0:
            self.signal_estimator = FilterSequence([self.signal_estimator, DelayFilter(int(self.fs*self.delay_ms/1000))])

        return self.signal_estimator

    def spatial_filter_is_zeros(self):
        return (self.spatial_filter == 0).all()

    def update(self, chunk):
        filtered_chunk = np.dot(chunk, self.spatial_matrix)
        current_chunk = self.signal_estimator.apply(filtered_chunk)
        if self.scaling_flag and self.std > 0:
            current_chunk = (current_chunk - self.mean) / self.std
        self.current_chunk = current_chunk
        return current_chunk

    def update_statistics(self, raw=None, emulate=False, signals_recorder=None, stats_type='meanstd'):
        if raw is not None and emulate:
            signal_recordings = np.zeros_like(signals_recorder[:, self.ind])
            mean_chunk_size = 8
            for k in range(0, raw.shape[0] - mean_chunk_size, mean_chunk_size):
                chunk = raw[k:k + mean_chunk_size]
                self.update(chunk)
                signal_recordings[k:k + mean_chunk_size] = self.current_chunk
        else:
            signal_recordings = signals_recorder[:, self.ind]
        if stats_type == 'meanstd':
            self.mean = signal_recordings.mean()
            self.std = signal_recordings.std()
        elif stats_type == 'max':
            self.std = signal_recordings.max()
            self.std = 1 if self.std == 0 else self.std
            self.mean = 0
        self.enable_scaling()
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
        self.signal_estimator = self.reset_signal_estimator()

    def drop_rejection(self, ind):
        self.rejections.drop(ind)
        self.update_spatial_filter()
        print(self.rejections)


    def enable_scaling(self):
        self.scaling_flag = True

    def descale_recording(self, data):
        return data * self.std + self.mean if self.scaling_flag else data

    def save_spatial_matrix(self, file_path, channels_labels=None):
        """
        Save full spatial matrix: R1*R2*...*Rk*S, where R1,..Rk - rejections matrices, S - spatial filter
        :return:
        """
        save_spatial_filter(file_path, self.spatial_matrix, channels_labels=channels_labels)
