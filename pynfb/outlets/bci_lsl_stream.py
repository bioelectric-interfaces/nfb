from time import time, sleep
from numpy.random import randint
from numpy import ones
from pylsl import StreamInfo, StreamOutlet


def simulate_bci_signal(fs, chunk_size=8, verbose=False):
    # setup stream
    info = StreamInfo(name='NFBLab_data', type='', channel_count=1, nominal_srate=fs)
    channels = info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("name", 'BCI')
    outlet = StreamOutlet(info)

    # prepare main loop
    start = time()
    counter = chunk_size
    x = x_base = ones((chunk_size, 1))

    # main loop
    while True:
        while time() - start < counter / fs:
            sleep(1 / fs)
        if randint(0, fs/chunk_size/2) == 0:
            x = x_base * randint(0, 2+1)
        outlet.push_chunk(x)
        counter += chunk_size
        if verbose:
            print('counter time: {:.2f}\ttime: {:.2f}'.format(counter/fs, time() - start))

if __name__ == '__main__':
    simulate_bci_signal(250, 8)