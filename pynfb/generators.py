import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet


def run_eeg_sim(n_channels=50, freq=500, chunk_size=0, source_buffer=None, name='example'):
    """
    Make LSL Stream Outlet and send source_buffer data or simulate sin data
    :param n_channels: number of channels
    :param freq: frequency
    :param chunk_size: chunk size
    :param source_buffer: buffer for samples to push (if it's None sine data will be sended)
    :param name: name of outlet
    :return:
    """
    # stream info
    info = StreamInfo(name=name, type='EEG', channel_count=n_channels, nominal_srate=500,
                      channel_format='float32', source_id='myuid34234')

    # channels labels (in accordance with XDF format, see also code.google.com/p/xdf)
    labels = ['Ch{}'.format(k + 1) for k in range(n_channels)]
    chns = info.desc().append_child("channels")
    for label in labels:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
    outlet = StreamOutlet(info, chunk_size=chunk_size)

    # send data and print some info every 5 sec
    print('now sending data...')
    t0 = time.time()
    t = t0
    c = 1
    ampl = 10
    freqs = np.arange(10, n_channels + 10)
    sample = np.zeros((n_channels,))
    while True:
        # if source_buffer is not None get sample from source_buffer
        # else simulate sin(a*t)*sin(b*t)
        if source_buffer is not None:
            sample[:source_buffer.shape[0]] = source_buffer[:, c % source_buffer.shape[1]]
        else:
            sample = np.sin(2 * np.pi * time.time() * 50) * 0 + np.sin(2 * np.pi * time.time() * freqs)
            sample *= (np.sin(2 * np.pi * time.time() * 0.25) + 1) * ampl
        # push sample end sleep 1/freq sec
        outlet.push_sample(sample)
        time.sleep(1. / freq)
        # print time, frequency and samples count every 5 sec
        if c % (freq * 5) == 0:
            t_curr = time.time()
            print('t={:.1f}, f={:.2f}, c={}'.format(t_curr - t0, 1. / (t_curr - t) * freq * 5, c))
            t = t_curr
        c += 1
    pass


if __name__ == '__main__':
    run_eeg_sim(chunk_size=0, name='NVX136_Data')
