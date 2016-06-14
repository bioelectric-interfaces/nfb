import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

ch_names = ['LPA', 'RPA', 'Nz', 'Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6',
            'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7',
            'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2',
            'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'Oz', 'O2', 'I1', 'Iz', 'I2', 'AFp9h', 'AFp7h', 'AFp5h', 'AFp3h',
            'AFp1h', 'AFp2h', 'AFp4h', 'AFp6h', 'AFp8h', 'AFp10h', 'AFF9h', 'AFF7h', 'AFF5h', 'AFF3h', 'AFF1h', 'AFF2h',
            'AFF4h', 'AFF6h', 'AFF8h', 'AFF10h', 'FFT9h', 'FFT7h', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h',
            'FFT8h', 'FFT10h', 'FTT9h', 'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'FTT8h',
            'FTT10h', 'TTP9h', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'TTP8h', 'TTP10h',
            'TPP9h', 'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h', 'PPO9h', 'PPO7h',
            'PPO5h', 'PPO3h', 'PPO1h', 'PPO2h', 'PPO4h', 'PPO6h', 'PPO8h', 'PPO10h', 'POO9h', 'POO7h', 'POO5h', 'POO3h',
            'POO1h', 'POO2h', 'POO4h', 'POO6h', 'POO8h', 'POO10h', 'OI1h', 'OI2h', 'Fp1h', 'Fp2h', 'AF9h', 'AF7h',
            'AF5h', 'AF3h', 'AF1h', 'AF2h', 'AF4h', 'AF6h', 'AF8h', 'AF10h', 'F9h', 'F7h', 'F5h', 'F3h', 'F1h', 'F2h',
            'F4h', 'F6h', 'F8h', 'F10h', 'FT9h', 'FT7h', 'FC5h', 'FC3h', 'FC1h', 'FC2h', 'FC4h', 'FC6h', 'FT8h',
            'FT10h', 'T9h', 'T7h', 'C5h', 'C3h', 'C1h', 'C2h', 'C4h', 'C6h', 'T8h', 'T10h', 'TP9h', 'TP7h', 'CP5h',
            'CP3h', 'CP1h', 'CP2h', 'CP4h', 'CP6h', 'TP8h', 'TP10h', 'P9h', 'P7h', 'P5h', 'P3h', 'P1h', 'P2h', 'P4h',
            'P6h', 'P8h', 'P10h', 'PO9h', 'PO7h', 'PO5h', 'PO3h', 'PO1h', 'PO2h', 'PO4h', 'PO6h', 'PO8h', 'PO10h',
            'O1h', 'O2h', 'I1h', 'I2h', 'AFp9', 'AFp7', 'AFp5', 'AFp3', 'AFp1', 'AFpz', 'AFp2', 'AFp4', 'AFp6', 'AFp8',
            'AFp10', 'AFF9', 'AFF7', 'AFF5', 'AFF3', 'AFF1', 'AFFz', 'AFF2', 'AFF4', 'AFF6', 'AFF8', 'AFF10', 'FFT9',
            'FFT7', 'FFC5', 'FFC3', 'FFC1', 'FFCz', 'FFC2', 'FFC4', 'FFC6', 'FFT8', 'FFT10', 'FTT9', 'FTT7', 'FCC5',
            'FCC3', 'FCC1', 'FCCz', 'FCC2', 'FCC4', 'FCC6', 'FTT8', 'FTT10', 'TTP9', 'TTP7', 'CCP5', 'CCP3', 'CCP1',
            'CCPz', 'CCP2', 'CCP4', 'CCP6', 'TTP8', 'TTP10', 'TPP9', 'TPP7', 'CPP5', 'CPP3', 'CPP1', 'CPPz', 'CPP2',
            'CPP4', 'CPP6', 'TPP8', 'TPP10', 'PPO9', 'PPO7', 'PPO5', 'PPO3', 'PPO1', 'PPOz', 'PPO2', 'PPO4', 'PPO6',
            'PPO8', 'PPO10', 'POO9', 'POO7', 'POO5', 'POO3', 'POO1', 'POOz', 'POO2', 'POO4', 'POO6', 'POO8', 'POO10',
            'OI1', 'OIz', 'OI2', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2']


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
    labels = ch_names[:n_channels]
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
