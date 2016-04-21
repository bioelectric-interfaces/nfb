import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream


def run_eeg_inlet(n_channels=30, freq=500):
    streams = resolve_stream('name', 'example')  # 'AudioCaptureWin')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0], max_buflen=1)
    t0 = time.time()
    t = t0
    c = 1
    while True:
        sample, timestamp = inlet.pull_sample()
        if c % (freq * 5) == 0:
            t_curr = time.time()
            print('t={:.1f}, f={:.2f}, sample.shape={}'.format(t_curr - t0,
                                                               1. / (t_curr - t) * freq * 5,
                                                               np.array(sample).shape))
            t = t_curr
        c += 1
    pass



def simple():
    print('blip blop')
    pass


def run_eeg_inlet_chunks(action=simple, n_channels=30, freq=120):
    streams = resolve_stream('name', 'example')  # 'AudioCaptureWin')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=8)
    t0 = time.time()
    t = t0
    c = 1
    chunk_counter = 0
    zero_counter = 0
    none_counter = 0
    while True:
        time.sleep(1. / freq)
        chunk, timestamp = inlet.pull_chunk()
        action()
        if chunk_counter is None:
            none_counter += 1
        if len(chunk)>0:
            if len(chunk)!=8:
                print(len(chunk))
            chunk_counter += 1
        else:
            zero_counter += 0
        #if len(chunk) > 0:
        if c % (freq) == 0:
            t_curr = time.time()
            print('t={:.1f}, f={:.2f}, chunks={}, zeros={}, nones={}'.format(t_curr - t0,
                                                         1. / (t_curr - t) * freq,
                                                         chunk_counter, zero_counter, none_counter))
            t = t_curr
            chunk_counter = 0
            zero_counter = 0
            none_counter = 0
        c += 1
    pass

if __name__ == '__main__':
    run_eeg_inlet_chunks()