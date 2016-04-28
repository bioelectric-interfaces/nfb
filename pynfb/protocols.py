import time, sys
from pynfb.lsl.widgets import *
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

class Protocol:
    def __init__(self, name='', duration=30):
        self.name = name
        self.duration = duration
        pass


class BaselineProtocol(Protocol):
    def __init__(self, duration=30):
        super().__init__(name='Baseline', duration=duration)
        pass


class FeedbackProtocol(Protocol):
    def __init__(self, duration=30):
        super().__init__(name='Feedback', duration=duration)
        pass


def main():
    p = FeedbackProtocol()


if __name__ == '__main__':
    main()