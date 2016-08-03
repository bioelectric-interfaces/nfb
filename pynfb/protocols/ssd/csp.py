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


def csp(x, fs, band, butter_order=3, regularization_coef=0.05):
    """
    """
    # apply filter
    x_filtered = butter_bandpass_filter(data=x, lowcut=band[0], highcut=band[1], fs=fs, order=butter_order)
    x_filtered_parts = [x_filtered[:x_filtered.shape[0] // 2], x_filtered[x_filtered.shape[0] // 2:]]
    cov_parts = []
    for part in x_filtered_parts:
        cov = np.dot(part.T, part) / part.shape[0]
        cov_parts.append(cov / np.trace(cov))

    # find filters
    regularization = lambda z: z + regularization_coef * np.trace(z) * np.eye(z.shape[0]) / z.shape[0]
    vals, vecs = eigh(regularization(cov_parts[0]), regularization(cov_parts[1]))
    vecs /= np.abs(vecs).max(0)

    # return vals, vecs and topographics (in descending order)
    reversed_slice = slice(-1, None, -1)
    topo = inv(vecs[:,reversed_slice]).T
    return vals[reversed_slice], vecs[:, reversed_slice], topo
