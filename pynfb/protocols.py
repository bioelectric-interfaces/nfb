import time, sys
from pynfb.lsl.widgets import *
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

class Protocol:
    def __init__(self, name='', duration=30):
        self.name = name
        self.duration = duration
        pass

    def run(self):
        t0 = time.time()
        actions_counter = 0
        while time.time() - t0 < self.duration:
            print('action', actions_counter)
            time.sleep(1)
            actions_counter += 1
        pass


class BaselineProtocol(Protocol):
    def __init__(self, duration=30):
        super().__init__(name='Baseline', duration=duration)
        pass


class FeedbackProtocol(Protocol):
    def __init__(self, duration=30):
        super().__init__(name='Feedback', duration=duration)
        pass

    def run(self):
        app = QtGui.QApplication(sys.argv)
        win = LSLCircleFeedbackWidget(n_samples=500, n_channels=1)
        timer = QtCore.QTimer()
        timer.timeout.connect(win.update)
        timer.start(1000 * 1. / win.plot_freq)
        sys.exit(app.exec_())
        pass


def main():
    p = FeedbackProtocol()
    p.run()


if __name__ == '__main__':
    main()