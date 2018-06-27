from time import time, sleep
from pylsl import StreamInfo, StreamOutlet


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
        outlet.push_sample([(counter%1001)/1000.0])
        counter += 1
        if verbose:
            if counter % fs == 0:
                print([(counter%1001)/1000.0])

if __name__ == '__main__':
    simulate_bci_signal(250., True)