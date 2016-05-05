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
        self.protocols = [BaselineProtocol(self.signals), FeedbackProtocol(self.signals), BaselineProtocol(self.signals)]
        self.current_protocol_ind = 0
        self.main = MainWindow(parent=None, current_protocol=self.protocols[self.current_protocol_ind])
        self.subject = self.main.subject_window
        self.stream = LSLStream()
        # timer
        main_timer = QtCore.QTimer(self)
        main_timer.timeout.connect(self.update)
        main_timer.start(1000 * 1. / 500)
        # protocol switcher timer
        self.protocols_timer = QtCore.QTimer(self)
        self.protocols_timer.timeout.connect(self.next_protocol)
        self.protocols_timer.start(1000 * 10)

        pass

    def update(self):
        chunk = self.stream.get_next_chunk()
        if chunk is not None:
            for i, signal in enumerate(self.signals):
                signal.update(chunk)
                self.current_samples[i] = signal.current_sample
            self.main.redraw_signals(self.current_samples[0], chunk)
            self.subject.update_protocol_state(self.current_samples[0], chunk_size=chunk.shape[0])

    def next_protocol(self):
        if self.current_protocol_ind < len(self.protocols)-1:
            print('protocol', self.current_protocol_ind, 'done')
            self.current_protocol_ind += 1
            self.subject.current_protocol = self.protocols[self.current_protocol_ind]
            self.subject.figure.clear()
            self.subject.current_protocol.widget_painter.prepare_widget(self.subject.figure)
            self.protocols_timer.stop()
            self.protocols_timer.start(1000 * 10)
        else:
            print('finish')
            self.protocols_timer.stop()

    def run(self):
        sys.exit(self.exec_())

if __name__=='__main__':
    experiment = Experiment()
    experiment.run()
