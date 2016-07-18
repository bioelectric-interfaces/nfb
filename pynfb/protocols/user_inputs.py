import numpy as np
from PyQt4 import QtGui, QtCore
from pynfb.generators import ch_names
from pynfb.protocols.ssd import TopomapSelector
from pynfb.widgets.helpers import ch_names_to_2d_pos


class SelectSSDFilterWidget(QtGui.QDialog):
    def __init__(self, data, pos, names=None, sampling_freq=500, parent=None):
        super(SelectSSDFilterWidget, self).__init__(parent)
        # layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        # label
        top_label = QtGui.QLabel('Select filter:')
        layout.addWidget(top_label)

        # topomap selector
        self.selector = TopomapSelector(data, pos, names=names, sampling_freq=sampling_freq)
        layout.addWidget(self.selector)

        # reject, select radio
        radio_layout = QtGui.QHBoxLayout()
        radio_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.select_radio = QtGui.QRadioButton('Select')
        self.select_radio.toggle()
        self.reject_radio = QtGui.QRadioButton('Reject')
        radio_layout.addWidget(self.select_radio)
        radio_layout.addWidget(self.reject_radio)
        layout.addLayout(radio_layout)

        # select button
        select_button = QtGui.QPushButton('OK')
        select_button.setMaximumWidth(100)
        select_button.clicked.connect(self.select_action)
        radio_layout.addWidget(select_button)

        # selected filter
        self.filter = self.selector.get_current_filter()

    def select_action(self):
        self.filter = self.selector.get_current_filter(reject=self.reject_radio.isChecked())
        self.bandpass = self.selector.get_current_bandpass()
        self.close()

    @staticmethod
    def select_filter_and_bandpass(data, pos, names=None, sampling_freq=500, parent=None):
        selector = SelectSSDFilterWidget(data, pos, names=names, sampling_freq=sampling_freq, parent=parent)
        _result = selector.exec_()
        return selector.filter, selector.bandpass

if __name__ == '__main__':
    app = QtGui.QApplication([])
    channels_names = ch_names[:128]
    x = np.random.rand(10000, 128)
    pos = ch_names_to_2d_pos(channels_names)
    widget = SelectSSDFilterWidget(x, pos, names=channels_names)
    widget.show()
    app.exec_()
