from pynfb.signals import DerivedSignal
from pynfb.lsl_stream import LSLStream, LSL_STREAM_NAMES
from PyQt4 import QtGui, QtCore
import sys


class Experiment():
    def __init__(self):
        self.stream = LSLStream()
        self.signal = DerivedSignal(bandpass_low=8, bandpass_high=12)
        pass

    def update(self):
        chunk = self.stream.get_next_chunk()
        if chunk is not None:
            self.signal.update(chunk)
            print(self.signal.current_sample)

    def run(self):
        app = QtGui.QApplication(sys.argv)
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(1000 * 1. / 500)
        sys.exit(app.exec_())

if __name__=='__main__':
    experiment = Experiment()
    experiment.run()
