from PyQt5 import QtGui, QtWidgets
from PyQt5 import QtCore, QtWidgets

class BCIFitWidget(QtWidgets.QWidget):
    fit_clicked = QtCore.pyqtSignal()
    def __init__(self, bci_signal, *args):
        super(BCIFitWidget, self).__init__(*args)
        label = QtWidgets.QLabel(bci_signal.name)

        # fit button
        fit_button = QtWidgets.QPushButton('Fit model')
        fit_button.clicked.connect(lambda: self.fit_clicked.emit())

        # set layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(label)
        layout.addWidget(fit_button)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])


    class BCISignalMock:
        def __init__(self):
            self.name = 'bci'

    w = BCIFitWidget(BCISignalMock())
    w.show()
    app.exec_()
