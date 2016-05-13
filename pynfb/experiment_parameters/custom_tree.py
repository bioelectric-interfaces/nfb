import sys
from PyQt4 import QtGui
from collections import OrderedDict

signals = [{'sSignalName': 'Signal1',
             'fBandpassLowHz': 1,
             'fBandpassHighHz': 10},
            {'sSignalName': 'Signal2',
             'fBandpassLowHz': 1,
             'fBandpassHighHz': 30}
            ]


class SignalsList(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        label = QtGui.QLabel('Signals:')
        self.list = QtGui.QListWidget(self)

        self.set_data()
        self.list.itemDoubleClicked.connect(self.item_double_clicked_event)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        self.setLayout(layout)
        self.show()

    def item_double_clicked_event(self, item):
        self.signals_dialogs[self.list.currentRow()].open()

    def set_data(self):
        self.list.clear()
        self.signals_dialogs = []
        for signal in signals:
            item = QtGui.QListWidgetItem(signal['sSignalName'])

            self.signals_dialogs.append(SignalDialog(self, signal_name=signal['sSignalName']))
            self.list.addItem(item)

class SignalDialog(QtGui.QDialog):
    def __init__(self, parent, signal_name='Signal'):
        super().__init__(parent)
        self.parent_list = parent
        self.setWindowTitle('Properties: '+signal_name)
        self.form_layout = QtGui.QFormLayout(self)
        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(signal_name)
        self.form_layout.addRow('&Name:', self.name)
        # bandpass
        self.bandpass_low = QtGui.QSpinBox()
        self.bandpass_low.setRange(0, 250)
        self.bandpass_low.setValue(0)
        self.form_layout.addRow('&Bandpass low [Hz]:', self.bandpass_low)
        self.bandpass_high = QtGui.QSpinBox()
        self.bandpass_high.setRange(0, 250)
        self.bandpass_high.setValue(250)
        self.form_layout.addRow('&Bandpass high [Hz]:', self.bandpass_high)
        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def open(self):
        self.set_data()
        super().open()

    def set_data(self):
        print(self.parent().list.currentRow())
        self.bandpass_low.setValue(signals[self.parent().list.currentRow()]['fBandpassLowHz'])
        self.bandpass_high.setValue(signals[self.parent().list.currentRow()]['fBandpassHighHz'])

    def save_and_close(self):
        signals[self.parent().list.currentRow()]['sSignalName'] = self.name.text()
        self.parent().set_data()
        self.close()


if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    window = SignalsList()
    window.show()
    sys.exit(app.exec_())