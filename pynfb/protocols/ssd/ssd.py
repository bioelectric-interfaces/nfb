from scipy.signal import butter, lfilter
from scipy.linalg import eigh, inv
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=axis)
    return y


def ssd(x, fs, bands, butter_order=3):
    """
    The code can be used in two modes. In the classical SSD mode
    when the filters are found to emphasize the spectral peak agains two
    flanker bands. In this case band is to be of dim [3 x 2] with the
    band[2,:] - representing the central band and the other two flanker bands
    another possible usage is to emphasize the peak power over the broadband
    EEG and then band is a 1 x 2 array specifying only the central band.
    """
    # apply filter
    cov_x_filtered = []
    for band in bands:
        x_filtered = butter_bandpass_filter(data=x, lowcut=band[0], highcut=band[1], fs=fs, order=butter_order)
        cov = np.dot(x_filtered.T, x_filtered) / x_filtered.shape[0]
        cov_x_filtered.append(cov)

    # compute band specific covariance matrices
    if len(bands) == 1:
        cov_peak = cov_x_filtered[0]
        cov_flankers = np.dot(x.T, x) / x.shape[0]
    elif len(bands) == 3:
        cov_peak = cov_x_filtered[0]
        cov_flankers = 0.5 * cov_x_filtered[0] + 0.5 * cov_x_filtered[2]
    else:
        raise ValueError('Wrong format for <band> argument')

    # find filters
    regularization_coef = 0.05
    regularization = lambda z: z + regularization_coef * np.trace(z) * np.eye(z.shape[0]) / z.shape[0]
    vals, vecs = eigh(regularization(cov_peak), regularization(cov_flankers))

    # return vals, vecs and topographics (in descending order)
    reversed_slice = slice(-1, None, -1)
    topo = inv(vecs[reversed_slice]).T  # TODO: transpose or not transpose: that is the question
    return vals[reversed_slice], vecs[reversed_slice], topo


def ssd_analysis(x, sampling_frequency, freqs, flanker_delta=2):
    freq_delta = freqs[1] - freqs[0]
    bands = [[[fc - freq_delta / 2 - flanker_delta, fc - freq_delta / 2],
              [fc - freq_delta / 2, fc + freq_delta / 2],
              [fc + freq_delta / 2, fc + freq_delta / 2 + flanker_delta]] for fc in freqs]
    major_vals = []
    topographies = []
    for band in bands:
        vals, vecs, topos = ssd(x, sampling_frequency, band)
        major_vals.append(vals[0])
        topographies.append(topos[:, 0])
    return major_vals, topographies


if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.rand(10000, 3)
    x[:, 1] += np.sin(np.arange(10000))*10
    print(ssd(np.random.rand(10000, 3), 500, [[9, 8]]))
    print('--')
    print(ssd_analysis(x, 500, np.arange(4,26), 3))