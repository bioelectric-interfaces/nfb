import sys
from PyQt5 import QtGui, QtWidgets

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt

import random

class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QtWidgets.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
