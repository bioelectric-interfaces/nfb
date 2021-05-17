from scipy.signal import butter, lfilter, filtfilt
from scipy.linalg import eigh, inv, eig
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, axis=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def ssd(x, fs, bands, butter_order=3, regularization_coef=0.05):
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
        cov_peak = cov_x_filtered[1]
        cov_flankers = 0.5 * cov_x_filtered[0] + 0.5 * cov_x_filtered[2]
    else:
        raise ValueError('Wrong format for <band> argument')

    # find filters
    regularization = lambda z: z + regularization_coef * np.trace(z) * np.eye(z.shape[0]) / z.shape[0]
    vals, vecs = eigh(regularization(cov_peak), regularization(cov_flankers))
    vecs /= np.abs(vecs).max(0)

    # like matlab
    # vals, vecs = eig(regularization(cov_peak), regularization(cov_flankers))
    # srt_key = np.argsort(vals)
    # vals = np.real(vals)[srt_key]
    # vecs = vecs[:, srt_key]

    # return vals, vecs and topographics (in descending order)
    reversed_slice = slice(-1, None, -1)
    topo = inv(vecs[:,reversed_slice]).T
    return vals[reversed_slice], vecs[:, reversed_slice], topo


def ssd_analysis(x, sampling_frequency, freqs, flanker_delta=2, flanker_margin=0, regularization_coef=0.05):
    freq_delta = freqs[1] - freqs[0]
    bands = [
        [[fc - flanker_delta - flanker_margin, fc - flanker_margin],
         [fc, fc + freq_delta],
         [fc + freq_delta + flanker_margin, fc + freq_delta + flanker_delta + flanker_margin]]
        for fc in freqs]
    major_vals = []
    topographies = []
    filters = []
    for band in bands:
        vals, vecs, topos = ssd(x, sampling_frequency, band, regularization_coef=regularization_coef)
        major_vals.append(vals)
        topographies.append(topos)
        filters.append(vecs)
    return np.array(major_vals), np.array(topographies), filters


if __name__ == "__main__":
    x = np.loadtxt('example_recordings.txt')[:, :]
    k, l, n = ssd(x, 1000, [[7,9], [9, 10], [10, 12]])
    print(l)