from numpy.fft import fft, ifft
import numpy as np
import pylab as plt
from scipy.signal import butter, filtfilt, hilbert, welch
fs = 500


alpha = filtfilt(*butter(1, [8/fs*2, 12/fs*2], btype='band'), np.random.normal(size=50000))
envelope = np.abs(hilbert(alpha))
plt.plot(alpha/envelope)
plt.plot(np.real(np.exp(1j*np.angle(hilbert(alpha)))))


aa = plt.figure()
f, a = welch(alpha/envelope , fs=500, nperseg=2000 )
plt.plot(f, a/max(a))
f, a = welch(alpha, fs=500, nperseg=2000)
plt.plot(f, a/max(a))
f, a = welch(np.imag(hilbert(alpha)), fs=500, nperseg=2000 )
plt.plot(f, a/max(a))
#ifft(np.log(np.abs(fft(frames2))))
plt.figure()
freq = np.fft.fftfreq(len(alpha), 1/fs)
#plt.plot(freq[freq>0], np.abs(np.log((fft(alpha))))[freq>0])
keps = np.real(ifft(np.log(fft(alpha))))


#filtering
keps_filtered = filtfilt(*butter(4, [12./fs*2], btype='low'), keps)

#
xx = np.real(ifft(np.exp(fft(keps_filtered))))

#keps[ np.abs(freq)>14] = 0
plt.plot(alpha)
plt.plot(xx)


f, a = welch(xx , fs=500, nperseg=2000 )
aa.gca().plot(f, a/max(a))

plt.show()