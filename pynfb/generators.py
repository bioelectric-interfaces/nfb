import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3', 'Cz',
            'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2',
            'Fpz', 'Af7', 'Af3', 'Af4', 'Af8', 'F5', 'F1', 'F2', 'F6', 'Ft7', 'Fc3', 'Fcz', 'Fc4', 'Ft8', 'C5', 'C1',
            'C2', 'C6', 'Tp7', 'Cp3', 'Cpz', 'Cp4', 'Tp8', 'P5', 'P1', 'P2', 'P6', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
            'Aff1h', 'Aff2h', 'F9', 'F10', 'Ffc5h', 'Ffc1h', 'Ffc2h', 'Ffc6h', 'Ftt7h', 'Fcc3h', 'Fcc4h', 'Ftt8h',
            'Ccp5h', 'Ccp1h', 'Ccp2h', 'Ccp6h', 'Tpp7h', 'Cpp3h', 'Cpp4h', 'Tpp8h', 'P9', 'P10', 'Ppo9h', 'Ppo1h',
            'Ppo2h', 'Ppo10h', 'Po9', 'Po10', 'I1', 'Oi1h', 'Oi2h', 'I2', 'Afp1', 'Afp2', 'Aff5h', 'Aff6h', 'Fft9h',
            'Fft7h', 'Ffc3h', 'Ffc4h', 'Fft8h', 'Fft10h', 'Ftt9h', 'Fcc5h', 'Fcc1h', 'Fcc2h', 'Fcc6h', 'Ftt10h',
            'Ttp7h', 'Ccp3h', 'Ccp4h', 'Ttp8h', 'Tpp9h', 'Cpp5h', 'Cpp1h', 'Cpp2h', 'Cpp6h', 'Tpp10h', 'Ppo5h', 'Ppo6h',
            'Poo9h', 'Poo1', 'Poo2', 'Poo10h', 'Aux 1.1', 'Aux 1.2', 'Aux 2.1', 'Aux 2.2', 'Aux 3.1', 'Aux 3.2',
            'Aux 4.1', 'Aux 4.2']

ch_names32 = ['Fp1','Fp2','F7','F3','Fz','F4','F8','Ft9','Fc5','Fc1','Fc2','Fc6','Ft10','T7','C3','Cz','C4','T8','Tp9',
              'Cp5','Cp1','Cp2','Cp6','Tp10','P7','P3','Pz','P4','P8','O1','Oz','O2']
# TODO: remove when reading from .fif is implemented
ch_names32 = ['Ch ' + str(i + 1) for i in range(376)]


def run_eeg_sim(freq=None, chunk_size=0, source_buffer=None, name='example', labels=None):
    """
    Make LSL Stream Outlet and send source_buffer data or simulate sin data
    :param n_channels: number of channels
    :param freq: frequency
    :param chunk_size: chunk size
    :param source_buffer: buffer for samples to push (if it's None sine data will be sended)
    :param name: name of outlet
    :return:
    """
    # default freq
    freq = freq or 500

    # labels and n_channels
    labels = labels or ch_names32
    n_channels = len(labels) if labels is not None else 32

    # stream info
    info = StreamInfo(name=name, type='EEG', channel_count=n_channels, nominal_srate=freq,
                      channel_format='float32', source_id='myuid34234')

    # channels labels (in accordance with XDF format, see also code.google.com/p/xdf)

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
    freqs = np.arange(n_channels)*5 + 10
    sample = np.zeros((n_channels,))
    if source_buffer is not None:
        source_buffer = np.concatenate([source_buffer.T, source_buffer.T[::-1]]).T

    while True:
        # if source_buffer is not None get sample from source_buffer
        # else simulate sin(a*t)*sin(b*t)
        if source_buffer is not None:
            sample[:source_buffer.shape[0]] = source_buffer[:, c % source_buffer.shape[1]]
        else:
            sample = np.sin(2 * np.pi * time.time() * 50) * 0 + np.sin(2 * np.pi * time.time() * freqs)
            # sample *= (np.sin(2 * np.pi * time.time() * 0.25) + 1) * ampl
            sample *= c % (500 * 4) * ampl

        if c % 20000 > 10000:
            sample[0] *= 1
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
