# https://pythonprogramminglanguage.com/pyqt5-hello-world/
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QFormLayout, QWidget
from PyQt5.QtCore import QSize
import time

import logging
# import parallel
from psychopy import parallel
import time

class HelloWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(640, 480))
        self.setWindowTitle("Hello world - pythonprogramminglanguage.com")

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        # gridLayout = QGridLayout(self)
        # centralWidget.setLayout(gridLayout)

        layout = QFormLayout(self)
        centralWidget.setLayout(layout)

        title = QLabel("Hello World from PyQt", self)
        title.setAlignment(QtCore.Qt.AlignCenter)
        # layout.addWidget(title, 0, 0)

        parallel_high = QtWidgets.QPushButton('Parallel_0_set_high')
        parallel_high.clicked.connect(lambda x: self.send_parallel_info(1))
        layout.addRow('', parallel_high)

        parallel_low = QtWidgets.QPushButton('Parallel_0_set_low')
        parallel_low.clicked.connect(lambda x: self.send_parallel_info(0))
        layout.addRow('', parallel_low)

        # Setup parallel port
        # self.p_port = parallel.Parallel(port=0x2010)
        self.p_port = parallel.ParallelPort(address=0x2010)

    def send_parallel_info(self, data):
        # send some info over the parallel port
        self.p_port.setData(data)
        time.sleep(0.05)
        self.p_port.setData(0)
        print(f'sending: {data}')

    def keyPressEvent(self, e):
        # TODO: make it so this can't be pressed when the pre-calcs are happening!!!!

        if e.key() == QtCore.Qt.Key_Space:
            # If the space key is pressed, then start the block
            # Get the timestamp the key was pressed
            # timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
            self.key_press_time = int(time.time() * 1000)
            print(f" SPACE PRESSED AT: {self.key_press_time}")

        if e.key() == QtCore.Qt.Key_Escape:
            # Toggle fullscreen on the escape key
            if self.windowState() & QtCore.Qt.WindowFullScreen:
                self.showNormal()
            else:
                self.showFullScreen()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = HelloWindow()
    mainWin.show()
    sys.exit( app.exec_() )