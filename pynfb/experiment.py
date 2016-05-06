from pynfb.signals import DerivedSignal
from pynfb.lsl_stream import LSLStream, LSL_STREAM_NAMES
from pynfb.protocols import BaselineProtocol, FeedbackProtocol
from PyQt4 import QtGui, QtCore
import sys
from pynfb.windows import  MainWindow
import numpy as np

class Experiment(QtGui.QApplication):
    def __init__(self):
        super(Experiment, self).__init__(sys.argv)
        # inlet frequency
        self.freq = 500
        # signals
        self.signals = [DerivedSignal(bandpass_low=8, bandpass_high=12), DerivedSignal(bandpass_low=40, bandpass_high=50)]
        self.current_samples = np.zeros_like(self.signals)
        # protocols
        self.protocols = [BaselineProtocol(self.signals, duration=10, base_signal_id=1),
                          FeedbackProtocol(self.signals, duration=10),
                          BaselineProtocol(self.signals, duration=10)]

        self.current_protocol_ind = 0
        self.main = MainWindow(parent=None, current_protocol=self.protocols[self.current_protocol_ind], n_signals=len(self.signals))
        self.subject = self.main.subject_window
        self.stream = LSLStream()
        # timer
        main_timer = QtCore.QTimer(self)
        main_timer.timeout.connect(self.update)
        main_timer.start(1000 * 1. / self.freq)
        # samples counter for protocol sequence
        self.samples_counter = 0
        self.current_protocol_n_samples = self.freq * self.protocols[self.current_protocol_ind].duration
        pass

    def update(self):
        chunk = self.stream.get_next_chunk()
        if chunk is not None:
            self.samples_counter += chunk.shape[0]
            for i, signal in enumerate(self.signals):
                signal.update(chunk)
                self.current_samples[i] = signal.current_sample
            self.main.redraw_signals(self.current_samples, chunk)
            self.subject.update_protocol_state(self.current_samples, chunk_size=chunk.shape[0])
            if self.samples_counter >= self.current_protocol_n_samples:
                self.next_protocol()

    def next_protocol(self):
        print('protocol:', self.current_protocol_ind, 'samples:', self.samples_counter)

        self.samples_counter = 0
        if self.current_protocol_ind < len(self.protocols)-1:
            self.protocols[self.current_protocol_ind].close_protocol()
            if self.protocols[self.current_protocol_ind].update_statistics_in_the_end:
                self.main.signals_buffer *= 0
            self.current_protocol_ind += 1
            self.current_protocol_n_samples = self.freq * self.protocols[self.current_protocol_ind].duration
            self.subject.change_protocol(self.protocols[self.current_protocol_ind])
        else:
            print('finish')
            self.current_protocol_n_samples = np.inf
            self.subject.close()

    def run(self):
        sys.exit(self.exec_())

if __name__=='__main__':
    experiment = Experiment()
    experiment.run()
