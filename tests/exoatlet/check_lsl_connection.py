from time import time, sleep
from pylsl import StreamInfo, StreamOutlet, resolve_streams


def simulate_bci_signal(fs, verbose=False):
    """
    :param fs: sampling frequency
    :param verbose: if verbose == True print info
    """

    # setup stream
    info = StreamInfo(name='NFBLab_data', type='', channel_count=1, nominal_srate=fs)
    channels = info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("name", 'BCI')
    print('Stream info:\n\tname: {}, fs: {}Hz, channels: {}'.format(info.name(), int(info.nominal_srate()), ['BCI']))
    outlet = StreamOutlet(info)
    print('Now sending data...')

    # prepare main loop
    start = time()
    counter = 0
    # main loop
    while True:
        while time() - start < counter / fs:
            sleep(1 / fs)

        x = 1-int((counter%1501)<250)
        outlet.push_sample([x])
        counter += 1
        if verbose:
            if counter % fs == 0:
                print(x)

if __name__ == '__main__':
    print([s.name() for s in resolve_streams()])
    simulate_bci_signal(250., True)