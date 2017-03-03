import numpy as np
import pylab as plt
from scipy.signal import hilbert, firwin2, filtfilt
from scipy.fftpack import rfft, irfft, fftfreq




def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = np.zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y

if __name__ == '__main__':
    fs = 500
    band = (9, 14)
    x = np.random.normal(size=(5000, ))
    plt.plot(x, alpha=0.2)
    w = fftfreq(x.shape[0], d=1. / fs * 2)
    f_signal = rfft(x)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < band[0]) | (w > band[1])] = 0
    cut_signal = irfft(cut_f_signal)
    plt.plot(cut_signal)
    print(np.sum(np.abs(x)**2), 2*np.sum(np.abs(f_signal)**2)/x.shape[0])
    print(np.sum(np.abs(cut_signal) ** 2), 2 * np.sum(np.abs(cut_f_signal) ** 2)/x.shape[0])
    print(np.sum(np.abs(hilbert(x)**2))/2, np.sum(np.abs(hilbert(cut_signal)**2))/2)
    plt.figure()
    plt.plot(w, cut_f_signal**2)
    plt.show()
