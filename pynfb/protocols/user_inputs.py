import numpy as np
from PyQt4 import QtGui, QtCore
from pynfb.generators import ch_names
from pynfb.protocols.ssd import TopomapSelector
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup


class SelectSSDFilterWidget(QtGui.QDialog):
    def __init__(self, data, pos, names=None, sampling_freq=500, parent=None):
        super(SelectSSDFilterWidget, self).__init__(parent)
        self.data = data
        self.rejections = []

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
        self.select_radio = QtGui.QPushButton('Select and Close')
        self.select_radio.clicked.connect(self.select_action)
        self.reject_radio = QtGui.QPushButton('Reject')
        self.reject_radio.clicked.connect(self.reject_data)
        radio_layout.addWidget(self.reject_radio)
        radio_layout.addWidget(self.select_radio)

        # update checkboxes layout
        update_layout = QtGui.QVBoxLayout()
        self.update_band_checkbox = QtGui.QCheckBox('Update band')
        self.update_band_checkbox.setChecked(True)
        self.update_filter_checkbox = QtGui.QCheckBox('Update spatial filter (from SSD)')
        self.update_filter_checkbox.setChecked(True)
        self.manual_filter_checkbox = QtGui.QCheckBox('Update spatial filter (manual)')
        self.manual_filter_checkbox.setChecked(False)

        def uncheck_if_true(checkbox, flag):
            if flag:
                checkbox.setChecked(False)
        self.manual_filter_checkbox.stateChanged.connect(lambda: uncheck_if_true(self.update_filter_checkbox,
                                                                                 self.manual_filter_checkbox.isChecked()))

        self.update_filter_checkbox.stateChanged.connect(lambda: uncheck_if_true(self.manual_filter_checkbox,
                                                                                 self.update_filter_checkbox.isChecked()))

        update_layout.addWidget(self.update_band_checkbox)
        update_layout.addWidget(self.update_filter_checkbox)
        update_layout.addWidget(self.manual_filter_checkbox)
        layout.addLayout(update_layout)
        layout.addLayout(radio_layout)

        # selected filter
        self.filter = self.selector.get_current_filter()



    def reject_data(self):
        rejection = self.selector.get_current_filter(reject=True)
        self.rejections.append(rejection)
        self.data = np.dot(self.data, self.selector.get_current_filter(reject=True))
        self.selector.update_data(self.data)
        self.selector.recompute_ssd()

    def select_action(self):
        self.filter = self.selector.get_current_filter()
        self.bandpass = self.selector.get_current_bandpass()
        self.close()

    @staticmethod
    def select_filter_and_bandpass(data, pos, names=None, sampling_freq=500, parent=None):
        selector = SelectSSDFilterWidget(data, pos, names=names, sampling_freq=sampling_freq, parent=parent)
        _result = selector.exec_()
        if selector.update_filter_checkbox.isChecked():
            filter = selector.filter
        elif selector.manual_filter_checkbox.isChecked():
            filter = SpatialFilterSetup.get_filter(names, message='Please modify spatial filter for current signal')
        else:
            filter = None
        return (filter,
                selector.bandpass if selector.update_band_checkbox.isChecked() else None,
                selector.rejections)

if __name__ == '__main__':
    import numpy as np
    from pynfb.widgets.helpers import ch_names_to_2d_pos

    app = QtGui.QApplication([])

    ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
                'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    channels_names = np.array(ch_names)
    x = np.random.randn(1000, len(ch_names)-1) #np.loadtxt('ssd/example_recordings.txt')[:, channels_names != 'Cz']

    channels_names = list(channels_names[channels_names != 'Cz'])
    pos = ch_names_to_2d_pos(channels_names)

    # double ssd reject test
    for _k in range(3):
        filter, bandpass, rejections = SelectSSDFilterWidget.select_filter_and_bandpass(x, pos, names=channels_names,
                                                                            sampling_freq=1000)
        # ff = np.zeros((filter.shape[0], filter.shape[0]))
        # ff[:, 0] = filter
        # filter = ff
        for rejection in rejections:
            x = np.dot(x, rejection)