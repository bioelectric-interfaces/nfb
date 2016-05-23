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


def run_eeg_sim(n_channels=50, freq=500, chunk_size=0, source_buffer=None):
    info = StreamInfo(name='example', type='EEG', channel_count=n_channels, nominal_srate=500,  # TODO: nominal_srate?
                      channel_format='float32', source_id='myuid34234')
    outlet = StreamOutlet(info, chunk_size=chunk_size)
    print('now sending data...')
    t0 = time.time()
    t = t0
    c = 1
    ampl = 1
    freqs = np.arange(10,n_channels+10)
    sample = np.zeros((n_channels, ))
    while True:
        if source_buffer is not None:
            sample[:source_buffer.shape[0]] = source_buffer[:, c%source_buffer.shape[1]]
        else:
            sample = np.random.uniform(-0.1, 0.1, size=(n_channels, ))*0
            sample += np.sin(2*np.pi*time.time()*50)*0 + np.sin(2*np.pi*time.time()*freqs)
            sample *= (np.sin(2*np.pi*time.time()*0.25)+1)*ampl
            if c % (freq * 5) < 10:
                sample += np.ones(shape=(n_channels, ))*5*ampl*0
        outlet.push_sample(sample)

        time.sleep(1./freq)
        if c%(freq*5)==0:
            t_curr = time.time()
            print('t={:.1f}, f={:.2f}, c={}'.format(t_curr - t0 ,1./(t_curr - t)*freq*5, c))
            t = t_curr
        c += 1
    pass

if __name__ == '__main__':
    source_buffer = None if 1 else np.load('E:\\_nikolai\\projects\\nfb\pynfb\\results\\raw.npy').T
    run_eeg_sim(chunk_size=0, source_buffer=source_buffer)