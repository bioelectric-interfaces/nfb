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
        # signals
        self.signals = [DerivedSignal(bandpass_low=8, bandpass_high=12)]
        self.current_samples = np.zeros_like(self.signals)
        # protocols
        self.protocols = [BaselineProtocol(self.signals), FeedbackProtocol(self.signals)]
        self.main = MainWindow(parent=None, current_protocol=self.protocols[1])
        self.subject = self.main.subject_window
        self.stream = LSLStream()
        # timer
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(1000 * 1. / 500)
        pass

    def update(self):

        chunk = self.stream.get_next_chunk()
        if chunk is not None:
            for i, signal in enumerate(self.signals):
                signal.update(chunk)
                self.current_samples[i] = signal.current_sample
            self.main.redraw_signals(self.current_samples[0], chunk)
            self.subject.update_protocol_state(self.current_samples[0], chunk_size=chunk.shape[0])

    def run(self):
        sys.exit(self.exec_())

if __name__=='__main__':
    experiment = Experiment()
    experiment.run()
