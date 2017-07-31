import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from  scipy import fftpack

def band_hilbert(x):
    N = len(x)
    Xf = fftpack.fft(x, N)


    w = fftpack.fftfreq(N, 1/fs)
    Xf[np.abs(w) <8] = 0
    Xf[np.abs(w) > 13] = 0
    #plt.plot(w, np.abs(Xf))
    #plt.show()

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return fftpack.ifft(Xf * h), Xf*h/np.sqrt(len(Xf))


duration = 11
fs = 1000.0
samples = int(fs*duration)
t = np.arange(samples) / fs
signal = chirp(t, 25.0, t[-1], 40.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))

signal2 = chirp(t, 25.0, t[-1], 40.0)
signal2 *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t - 0))


phase = np.linspace(-np.pi, np.pi, 1000)
x_list = []
y_list = []
for ph in phase:
    signal  = np.sin(2.0*np.pi*10.0*t - 0)
    signal2 = np.sin(2.0*np.pi*(10)*t + ph)
    analytic_signal, xf = band_hilbert(signal)
    analytic_signal2, xf2 = band_hilbert(signal2)
    x_list.append(np.imag(np.dot(analytic_signal2, analytic_signal.conj()))/np.sqrt(np.abs(np.dot(analytic_signal, analytic_signal.conj())*np.dot(analytic_signal2, analytic_signal2.conj()))))
    y_list.append(np.imag(np.dot(xf, xf2.conj()))/np.sqrt(np.abs(np.dot(xf, xf.conj())*np.dot(xf2, xf2.conj()))))

print(np.array(x_list)/np.array(y_list))

#plt.plot(phase, x_list)
plt.plot(phase, y_list)
plt.show()
#analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)

plt.show()


