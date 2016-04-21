import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet


def run_noise_sins():
    channel_count = 1
    info = StreamInfo(name='example', type='EEG', channel_count=channel_count, nominal_srate=1, channel_format='float32',
                      source_id='myuid34234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    freqs = np.linspace(0, 20, channel_count)
    while True:
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        mysample = np.random.randn(channel_count) + np.sin(2*np.pi*time.time()*freqs)*5
        # now send it and wait for a bit
        outlet.push_sample(mysample)
        time.sleep(0.002)
    pass


def run_eeg_sim(n_channels=30, freq=500, chunk_size=0):
    info = StreamInfo(name='example', type='EEG', channel_count=n_channels, nominal_srate=500,  # TODO: nominal_srate?
                      channel_format='float32', source_id='myuid34234')
    outlet = StreamOutlet(info, chunk_size=chunk_size)
    print('now sending data...')
    t0 = time.time()
    t = t0
    c = 1
    while True:
        outlet.push_sample(np.random.uniform(size=(n_channels, )))
        time.sleep(1./freq)
        if c%(freq*5)==0:
            t_curr = time.time()
            print('t={:.1f}, f={:.2f}'.format(t_curr - t0 ,1./(t_curr - t)*freq*5))
            t = t_curr
        c += 1
    pass

if __name__ == '__main__':
    run_eeg_sim(chunk_size=0)