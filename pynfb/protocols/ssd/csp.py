from scipy.signal import butter, lfilter, filtfilt
from scipy.linalg import eigh, inv, eig
from scipy.fftpack import rfft, irfft, fftfreq
import numpy as np


def fft_filter(x, fs, band=(9, 14)):
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x, axis=0)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal, axis=0)
    return cut_signal

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


def csp3(x_dict, fs, band, butter_order=6, regularization_coef=0.01, lambda_=0.5):
    """
    """
    if not isinstance(x_dict, dict):
        n = x_dict.shape[0]
        x_dict = {
            'closed': x_dict[:n//3],
            'opened': x_dict[n//3:2*n//3],
            'rotate': x_dict[2*n//3:]
        }
    # apply filter
    cov_dict = {}
    for key, x in x_dict.items():
        x_filtered = x#fft_filter(x, fs, band)
        cov_dict[key] = np.dot(x_filtered.T, x_filtered) / x_filtered.shape[0]
        cov_dict[key] /= np.trace(cov_dict[key])

    # find filters
    regularization = lambda z: z + regularization_coef * np.eye(z.shape[0]) * np.trace(z)
    R1 = cov_dict['opened']
    R2 = (1-lambda_)*(cov_dict['closed'] - cov_dict['opened']) + lambda_*cov_dict['rotate']
    #print(R2)
    vals, vecs = eigh(regularization(R1), regularization(R2))
    #print(vals)
    vecs /= np.abs(vecs).max(0)

    # return vals, vecs and topographics (in descending order)
    reversed_slice = slice(-1, None, -1)
    topo = inv(vecs[:,reversed_slice]).T
    return vals[reversed_slice], vecs[:, reversed_slice], topo

def csp(x, fs, band, butter_order=3, regularization_coef=0.05, lambda_=None):
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


def csp_new(state1, state2=None, reg_coef=0.05):
    states = [state1, state2] if state2 is None else np.split(state1, [state1.shape[0] // 2])
    covs = [np.dot(state.T, state) / state.shape[0] for state in states]

    # find filters
    regularization = lambda z: z + reg_coef * np.trace(z) * np.eye(z.shape[0]) / z.shape[0]
    vals, vecs = eigh(regularization(covs[0]), regularization(covs[1]))
    vecs /= np.abs(vecs).max(0)

    # return vals, vecs and topographics (in descending order)
    reversed_slice = slice(-1, None, -1)
    topo = inv(vecs[:, reversed_slice]).T
    return vals[reversed_slice], vecs[:, reversed_slice], topo