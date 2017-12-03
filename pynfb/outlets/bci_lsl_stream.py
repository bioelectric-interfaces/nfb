from time import time, sleep
import numpy as np
from pylsl import StreamInfo, StreamOutlet


def simulate_bci_signal(fs, chunk_size=8, verbose=False):
    # setup stream
    info = StreamInfo(name='NFBLab_data', type='', channel_count=1, nominal_srate=fs)
    channels = info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("name", 'BCI')
    print('Stream info:\n\tname: {}, fs: {}Hz, channels: {}'.format(info.name(), int(info.nominal_srate()), ['BCI']))
    outlet = StreamOutlet(info)
    print('Now sending data...')

    # prepare main loop
    start = time()
    counter = chunk_size
    x = x_base = np.ones((chunk_size, 1))
    n_chunks = 0

    # main loop
    while True:
        while time() - start < counter / fs:
            sleep(1 / fs)
        if np.random.randint(0, fs/chunk_size/2) == 0:
            x = x_base * np.random.randint(0, 2+1)
        outlet.push_chunk(x)
        with open("bci_current_state.pkl", "w") as fp:
            fp.write(str(np.random.randint(0, 2+1)))
        n_chunks += 1
        counter += chunk_size
        if verbose:
            # print('counter time: {:.2f}\ttime: {:.2f}'.format(counter/fs, time() - start))
            if n_chunks % 50 == 0:
                print('Chunk {} was sent'.format(n_chunks))

if __name__ == '__main__':
    simulate_bci_signal(250, 8, True)