import numpy as np
import time
from threading import Thread
import os


class LSLSim:
    def __init__(self, stream_name, fs, df, chunk_size):
        self.stream_name = stream_name
        self.fs = fs
        self.channels = [col for col in df.columns if col[0].isupper()]
        self.states_names = df.block_name.unique().tolist()
        self.df = df
        self.chunk_size = chunk_size

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
            print(len(data), self.current_block)
        self.total_samples += self.chunk_size
        # push chunk
        # print('push chunk {}x{}, current sample {}'.format(*chunk.shape, self.current_sample))

    def set_state_by_name(self, state_name):
        self.current_state = self.states_names.index(state_name)
        self.current_sample = 0
        self.state_block_numbers = self._get_block_numbers()
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
    try:
        from google_drive_downloader import GoogleDriveDownloader as gdd
    except ModuleNotFoundError as e:
        print(str(e) + '\nPlease install googledrivedownloader package:\n\n\tpip install googledrivedownloader')
        os._exit(0)
    import pandas as pd

    data_path = './utils/df_motor_probes.pkl'
    if not os.path.exists(data_path):
        gdd.download_file_from_google_drive(file_id='1O9PZ7TUnhcXpsNu3o4yRsVThofFWFuVP', dest_path=data_path)
    df = pd.read_pickle(data_path)
    fs = 250
    stream_name = 'lsl_sim'
    chunk_size = 125

    lsl_sim = LSLSim(stream_name, fs, df, chunk_size)

    break_flag = False
    def run_lsl_sim(n_seconds=20):
        print('{} Start streaming'.format(lsl_sim.time_str()))
        for k in range(n_seconds*fs//chunk_size):
            lsl_sim.push_chunk()
            time.sleep(chunk_size/fs)
            if break_flag: break
        print(lsl_sim.info())
        print('{} Stop streaming'.format(lsl_sim.time_str()))
        print('... Press any key to exit ...')

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