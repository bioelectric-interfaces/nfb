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
info = StreamInfo('python', 'EEG', 2)

# next make an outlet
outlet = StreamOutlet(info)

from pylsl import StreamInlet, resolve_stream
print('resolving stream')
streams = resolve_stream('name', 'matlab')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
print('resolved')

t = 0
mean_time = 0
while True:
    #time.sleep(0.002)
    t += 1
    clock = local_clock()
    outlet.push_sample([0, 1])
    sample, timestamp = inlet.pull_sample(timeout=1)
    dt = local_clock() - clock
    mean_time += dt
    print(mean_time / t, dt)
    #time.sleep(0.001)