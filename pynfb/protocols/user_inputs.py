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

        # update checkboxes layout
        update_layout = QtGui.QVBoxLayout()
        self.update_band_checkbox = QtGui.QCheckBox('Update band')
        self.update_band_checkbox.setChecked(True)
        self.update_spatial_filter_checkbox = QtGui.QCheckBox('Update spatial filter')
        self.update_spatial_filter_checkbox.setChecked(True)
        update_layout.addWidget(self.update_band_checkbox)
        update_layout.addWidget(self.update_spatial_filter_checkbox)
        layout.addLayout(update_layout)

        # select button
        select_button = QtGui.QPushButton('OK')
        select_button.setMaximumWidth(100)
        select_button.clicked.connect(self.select_action)
        radio_layout.addWidget(select_button)
        layout.addLayout(radio_layout)

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
        return (selector.filter if selector.update_spatial_filter_checkbox.isChecked() else None,
                selector.bandpass if selector.update_band_checkbox.isChecked() else None)

if __name__ == '__main__':
    import numpy as np
    from pynfb.widgets.helpers import ch_names_to_2d_pos

    app = QtGui.QApplication([])

    ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
                'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    channels_names = np.array(ch_names)
    x = np.loadtxt('ssd/example_recordings.txt')[:, channels_names != 'Cz']
    channels_names = list(channels_names[channels_names != 'Cz'])
    pos = ch_names_to_2d_pos(channels_names)

    # double ssd reject test
    for _k in range(3):
        filter, bandpass = SelectSSDFilterWidget.select_filter_and_bandpass(x, pos, names=channels_names,
                                                                            sampling_freq=1000)
        # ff = np.zeros((filter.shape[0], filter.shape[0]))
        # ff[:, 0] = filter
        # filter = ff
        x = np.dot(x, filter)