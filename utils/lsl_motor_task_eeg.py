import numpy as np
import time
import os
from pylsl import StreamInfo, StreamOutlet
# stream info

class MotorEEGStreamOutlet(StreamOutlet):
    def __init__(self, name, fs, n_channels, labels, chunk_size):
        # create stream info
        info = StreamInfo(name=name, type='EEG', channel_count=n_channels, nominal_srate=fs,
                          channel_format='float32', source_id='myuid34234')

        # set channels labels (in accordance with XDF format, see also code.google.com/p/xdf)
        chns = info.desc().append_child("channels")
        for label in labels:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)

        # init stream
        super(MotorEEGStreamOutlet, self).__init__(info, chunk_size=chunk_size)


class LSLSim:
    def __init__(self, stream_name, fs, df, chunk_size):
        self.stream_name = stream_name
        self.fs = fs
        self.channels = [col for col in df.columns if col[0].isupper()]
        self.states_names = df.block_name.unique().tolist()
        self.df = df
        self.chunk_size = chunk_size

        self.stream = MotorEEGStreamOutlet(stream_name, fs, len(self.channels), self.channels, chunk_size)

        self.current_state = 0
        self.current_sample = 0
        self.total_samples = 0
        self.state_block_numbers = self._get_block_numbers()
        self.current_block = self._pick_random_block()
        self.next_block = self._pick_random_block()

    def _get_block_numbers(self):
        return self.df.query('block_name=="{}"'.format(self.state))['block_number'].unique()

    def _pick_random_block(self):
        return self.state_block_numbers[np.random.randint(0, len(self.state_block_numbers))]

    def _get_block_data(self, block_number):
        return self.df.query('block_number=={}'.format(block_number))[self.channels].values

    def push_chunk(self):
        # prepare chunk
        data = self._get_block_data(self.current_block)
        if self.current_sample + self.chunk_size < len(data):
            chunk = data[self.current_sample:self.current_sample+self.chunk_size]
            self.current_sample += self.chunk_size
        else:
            data = self._get_block_data(self.next_block)
            self.current_block = self.next_block
            self.next_block = self._pick_random_block()
            chunk = data[:self.chunk_size]
            self.current_sample = self.chunk_size
            # print(len(data), self.current_block)
        self.stream.push_chunk(chunk.tolist())
        self.total_samples += self.chunk_size
        # print('push chunk {}x{}, current sample {}'.format(*chunk.shape, self.current_sample))

    def set_state_by_name(self, state_name):
        self.current_state = self.states_names.index(state_name)
        self.current_sample = 0
        self.state_block_numbers = self._get_block_numbers()
        self.current_block = self._pick_random_block()
        self.next_block = self._pick_random_block()
        print('{} Current state changed to {} {}'.format(self.time_str(), self.current_state, state_name))

    def info(self):
        info = self.time_str()
        info += " Current state: {} {}. Block #{}. ".format(self.current_state, self.state, self.current_block)
        info += "Sent {} samples.".format(self.total_samples, self.total_samples//self.fs)
        return info

    @property
    def state(self):
        return self.states_names[self.current_state]

    @state.setter
    def state(self, value):
        self.set_state_by_name(value)

    def time_str(self):
        _time_str = '{:02d}:{:02d}:{:02d}'.format(self.total_samples // 360 // self.fs,
                                      self.total_samples // 60 // self.fs % 60,
                                      self.total_samples // self.fs % 60)
        return _time_str


if __name__ == '__main__':
    GUI = True

    try:
        from google_drive_downloader import GoogleDriveDownloader as gdd
    except ModuleNotFoundError as e:
        print(str(e) + '\nPlease install googledrivedownloader package:\n\n\tpip install googledrivedownloader')
        os._exit(0)
    import pandas as pd
    import scipy.signal as sg

    data_path = './utils/df_motor_probes.pkl'
    if not os.path.exists(data_path):
        gdd.download_file_from_google_drive(file_id='1O9PZ7TUnhcXpsNu3o4yRsVThofFWFuVP', dest_path=data_path)

    df = pd.read_pickle(data_path)
    channels = [col for col in df.columns if col[0].isupper()]
    fs = 250
    df[channels] = sg.filtfilt(*sg.butter(3, 1 / fs * 2, 'high'), df[channels].values, axis=0)

    stream_name = 'lsl_sim'
    chunk_size = 25

    lsl_sim = LSLSim(stream_name, fs, df, chunk_size)

    # command line interface

    if not GUI:
        break_flag = False
        def run_lsl_sim(n_seconds=10):
            print('{} Start streaming'.format(lsl_sim.time_str()))
            for k in range(n_seconds * fs // chunk_size):
                lsl_sim.push_chunk()
                time.sleep(chunk_size / fs)
                if break_flag: break
            del lsl_sim.stream
            print(lsl_sim.info())
            print('{} Stop streaming'.format(lsl_sim.time_str()))
            print('... Press any key to exit ...')
        from threading import Thread
        thread = Thread(target=run_lsl_sim, daemon=True)
        thread.start()
        print(lsl_sim.info())
        while thread.is_alive():
            command = input('{} Enter state name or command "/stop", "/info":\n'.format(lsl_sim.time_str()))
            if not thread.is_alive(): break
            if command in lsl_sim.states_names:
                lsl_sim.state = command
            elif command == '/stop':
                break_flag = True
                thread.join()
                input()
            elif command == '/info':
                print(lsl_sim.info())
            else:
                print("{} State name or command \"{}\" doesn't exist".format(lsl_sim.time_str(), command))
            time.sleep(0.5)
        if not break_flag:
            thread.join()

    else:
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import QTimer
        app = QApplication([])
        window = QWidget()
        states_widget = QWidget()
        main_layout = QVBoxLayout()
        layout = QHBoxLayout()
        button = QPushButton('Rest')
        button.pressed.connect(lambda: lsl_sim.set_state_by_name('Rest'))
        layout.addWidget(button)
        button = QPushButton('Left')
        button.pressed.connect(lambda: lsl_sim.set_state_by_name('Left'))
        layout.addWidget(button)
        button = QPushButton('Right')
        button.pressed.connect(lambda: lsl_sim.set_state_by_name('Right'))
        layout.addWidget(button)
        button = QPushButton('Legs')
        button.pressed.connect(lambda: lsl_sim.set_state_by_name('Legs'))
        layout.addWidget(button)
        states_widget.setLayout(layout)
        window.setLayout(main_layout)
        main_layout.addWidget(states_widget)
        text = QLabel('Wow')
        main_layout.addWidget(text)
        def step():
            lsl_sim.push_chunk()
            text.setText(lsl_sim.info())
        window.show()
        timer = QTimer()
        timer.timeout.connect(step)
        timer.start(chunk_size/fs*1000)
        print('{} Start streaming'.format(lsl_sim.time_str()))
        def on_close():
            del lsl_sim.stream
            print(lsl_sim.info())
            print('{} Stop streaming'.format(lsl_sim.time_str()))
        app.aboutToQuit.connect(on_close)
        app.exec_()

