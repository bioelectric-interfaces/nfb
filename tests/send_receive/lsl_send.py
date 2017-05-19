"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np

# first create a new stream info (here we set the name to BioSemi,
# the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
# last value would be the serial number of the device or some other more or
# less locally unique identifier for the stream as far as available (you
# could also omit it but interrupted connections wouldn't auto-recover)
fs = 1000
info = StreamInfo('BioSemi', 'EEG', 2)

# next make an outlet
outlet = StreamOutlet(info)

print("now sending data...")
n_samples = fs * 100
recorder = np.zeros((n_samples, 2))

t = 0
t_start = local_clock()
while True:
    if t > n_samples - 2:
        break
    clock = local_clock()
    if clock - t_start > (t+1)/fs:
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        second = int(clock)
        mysample = [clock, int(second%5 == 0) if second - t_start > 5 else -1]
        t += 1
        outlet.push_sample(mysample, clock)
        recorder[t] = mysample

print(local_clock() - t_start)
np.save('sent_data.npy', recorder)