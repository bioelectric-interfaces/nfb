from PyQt5 import QtCore, QtGui, QtWidgets


class ChannelTroubleWarning(QtWidgets.QDialog):
    pause_clicked = QtCore.pyqtSignal()
    continue_clicked = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    def __init__(self, channels=None, parent=None):
        super().__init__(parent)
        # title
        self.setWindowTitle('Warning')

        # message
        if channels is None:
            self.message = 'Channels trouble detected.'
        else:
            if isinstance(channels, list):
                self.message = 'Trouble detected in channels {}.'.format(', '.join(channels))
            else:
                raise TypeError('channels should be list or None')
        self.message += ' You can pause the experiment and fix the problem or just continue.'

        # widgets
        label = QtWidgets.QLabel(self.message)
        self.ignore_flag = False
        self.ignore_checkbox = QtWidgets.QCheckBox("Don't show this warning again")
        self.continue_button = QtWidgets.QPushButton('Continue')
        self.pause_button = QtWidgets.QPushButton('Pause')

        # buttons handlers
        self.pause_button.clicked.connect(self.handle_pause_button)
        self.continue_button.clicked.connect(self.handle_continue_button)

        # layouts
        v_layout = QtWidgets.QVBoxLayout(self)
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.setAlignment(QtCore.Qt.AlignRight)

        # widgets drawing
        v_layout.addWidget(label)
        v_layout.addWidget(self.ignore_checkbox)
        v_layout.addLayout(h_layout)
        h_layout.addWidget(self.pause_button)
        h_layout.addWidget(self.continue_button)

    def handle_pause_button(self):
        self.pause_button.setDisabled(True)
        self.pause_clicked.emit()

    def handle_continue_button(self):
        self.continue_clicked.emit()
        self.close()

    def closeEvent(self, QCloseEvent):
        self.ignore_flag = self.ignore_checkbox.isChecked()
        self.closed.emit()
        super().closeEvent(QCloseEvent)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    w = ChannelTroubleWarning(['Cp', 'Cz'])
    w.pause_clicked.connect(lambda: print('pause clicked'))
    w.exec_()
    print(w.ignore_flag)
