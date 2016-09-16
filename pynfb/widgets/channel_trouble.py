from PyQt4 import QtGui, QtCore


class ChannelTroubleWarning(QtGui.QDialog):
    pause_clicked = QtCore.pyqtSignal()

    def __init__(self, channels=None, **kwargs):
        super().__init__(**kwargs)
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
        label = QtGui.QLabel(self.message)
        self.ignore_flag = False
        self.ignore_checkbox = QtGui.QCheckBox("Don't show this warning again")
        continue_button = QtGui.QPushButton('Continue')
        self.pause_button = QtGui.QPushButton('Pause')

        # buttons handlers
        self.pause_button.clicked.connect(self.handle_pause_button)
        continue_button.clicked.connect(self.close)

        # layouts
        v_layout = QtGui.QVBoxLayout(self)
        h_layout = QtGui.QHBoxLayout()
        h_layout.setAlignment(QtCore.Qt.AlignRight)

        # widgets drawing
        v_layout.addWidget(label)
        v_layout.addWidget(self.ignore_checkbox)
        v_layout.addLayout(h_layout)
        h_layout.addWidget(self.pause_button)
        h_layout.addWidget(continue_button)

    def handle_pause_button(self):
        self.pause_button.setDisabled(True)
        self.pause_clicked.emit()

    def closeEvent(self, QCloseEvent):
        self.ignore_flag = self.ignore_checkbox.isChecked()
        super().closeEvent(QCloseEvent)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = ChannelTroubleWarning(['Cp', 'Cz'])
    w.pause_clicked.connect(lambda: print('pause clicked'))
    w.exec_()
    print(w.ignore_flag)
