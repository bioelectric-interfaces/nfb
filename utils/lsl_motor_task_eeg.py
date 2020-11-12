import numpy as np
import time
from threading import Thread

class LSLSim:
    def __init__(self, stream_name, fs, channels, states, data, chunk_size):
        self.stream_name = stream_name
        self.fs = fs
        self.channels = channels
        self.states_names = states
        self.data = data
        self.chunk_size = chunk_size

        self.current_state = 0
        self.current_sample = 0
        self.total_samples = 0

    def push_chunk(self):
        # prepare chunk
        data = self.data[self.current_state]
        if self.current_sample + self.chunk_size < len(data):
            chunk = data[self.current_sample:self.current_sample+self.chunk_size]
            self.current_sample += self.chunk_size
        else:
            chunk = data[:self.chunk_size]
            self.current_sample = self.chunk_size
        self.total_samples += self.chunk_size
        # push chunk
        # print('push chunk {}x{}, current sample {}'.format(*chunk.shape, self.current_sample))

    def set_state_by_name(self, state_name):
        self.current_state = self.states_names.index(state_name)
        self.current_sample = 0
        print('{} Current state changed to {} {}'.format(self.time_str(), self.current_state, state_name))

    def info(self):
        info = self.time_str()
        info += " Current state: {} {}. ".format(self.current_state, self.state)
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
    channels = ['Fp1', 'Fp2', 'Oz']
    fs = 500
    stream_name = 'lsl_sim'
    states = ['Rest', 'Motor']
    data = [np.random.randn(fs * 10, len(channels)), np.random.randn(fs * 11, len(channels))]
    chunk_size=100

    lsl_sim = LSLSim(stream_name, fs, channels, states, data, chunk_size)

    break_flag = False
    def run_lsl_sim(n_seconds=10):
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