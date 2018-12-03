import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .._titles import WAIT_BAR_MESSAGES
from ..generators import ch_names
from ..protocols.ssd import TopomapSelector
from ..signal_processing.filters import SpatialRejection
from ..widgets.helpers import ch_names_to_2d_pos, WaitMessage
from ..widgets.spatial_filter_setup import SpatialFilterSetup


class SelectSSDFilterWidget(QtWidgets.QDialog):
    def __init__(self, data, pos, names=None, sampling_freq=500, parent=None, selector_class=TopomapSelector):
        super(SelectSSDFilterWidget, self).__init__(parent)
        self.data = data
        self.rejections = []
        self.topography = None

        # layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # label
        top_label = QtWidgets.QLabel('Select filter:')
        layout.addWidget(top_label)

        # topomap selector
        self.selector = selector_class(data, pos, names=names, sampling_freq=sampling_freq)
        layout.addWidget(self.selector)

        # reject, select radio
        radio_layout = QtWidgets.QHBoxLayout()
        radio_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.select_radio = QtWidgets.QPushButton('Select and Close')
        self.select_radio.clicked.connect(self.select_action)
        self.reject_radio = QtWidgets.QPushButton('Reject')
        self.reject_radio.clicked.connect(self.reject_data)
        radio_layout.addWidget(self.reject_radio)
        radio_layout.addWidget(self.select_radio)

        # update checkboxes layout
        update_layout = QtWidgets.QVBoxLayout()
        self.update_band_checkbox = QtWidgets.QCheckBox('Update band')
        self.update_band_checkbox.setChecked(selector_class==TopomapSelector)
        self.update_filter_checkbox = QtWidgets.QCheckBox(
            'Update spatial filter (from {})'.format('SSD' if selector_class==TopomapSelector else 'CSP'))
        self.update_filter_checkbox.setChecked(selector_class==TopomapSelector)
        self.manual_filter_checkbox = QtWidgets.QCheckBox('Update spatial filter (manual)')
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
        self.rejections.append(SpatialRejection(rejection, rank=1, type_str='ssd',
                                                topographies=self.selector.get_current_topo()))
        self.data = np.dot(self.data, self.selector.get_current_filter(reject=True))
        self.selector.update_data(self.data)
        self.selector.recompute()

    def select_action(self):
        self.filter = self.selector.get_current_filter()
        self.topography = self.selector.get_current_topo()
        self.bandpass = self.selector.get_current_bandpass()
        self.accept()
        self.close()

    @classmethod
    def select_filter_and_bandpass(cls, data, pos, names=None, sampling_freq=500, parent=None):
        wait_bar = WaitMessage(WAIT_BAR_MESSAGES['SSD']).show_and_return()
        selector = cls(data, pos, names=names, sampling_freq=sampling_freq, parent=parent)
        wait_bar.close()

        result = selector.exec_()

        # if window closed, return nothing
        if result == 0:
            return None, None, []

        # if select button was pressed, return filter (or None if corresponding checkbox was not checked),
        # bandpass (or None if corresponding checkbox was not checked) and all added rejections
        if selector.update_filter_checkbox.isChecked():
            filter = selector.filter
        elif selector.manual_filter_checkbox.isChecked():
            filter = SpatialFilterSetup.get_filter(names, message='Please modify spatial filter for current signal')
        else:
            filter = None
        return (filter, selector.topography,
                selector.bandpass if selector.update_band_checkbox.isChecked() else None,
                selector.rejections)


if __name__ == '__main__':
    import numpy as np
    from ..widgets.helpers import ch_names_to_2d_pos

    app = QtWidgets.QApplication([])

    ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
                'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    channels_names = np.array(ch_names)
    x = np.random.randn(1000, len(ch_names)-1) #np.loadtxt('ssd/example_recordings.txt')[:, channels_names != 'Cz']

    channels_names = list(channels_names[channels_names != 'Cz'])

    channels_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3',
                      'Cz',
                      'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz',
                      'O2']
    from ..serializers.hdf5 import load_h5py

    x = load_h5py('C:\\Users\\Nikolai\Downloads\\raw_.h5', 'protocol1')
    y = load_h5py('C:\\Users\\Nikolai\Downloads\\raw_.h5', 'protocol2')
    x = np.vstack((x[:y.shape[0]], y))
    pos = ch_names_to_2d_pos(channels_names)
    fs = 500
    # double ssd reject test
    for _k in range(3):
        filter, bandpass, rejections = SelectCSPFilterWidget.select_filter_and_bandpass(x, pos, names=channels_names,
                                                                            sampling_freq=fs)
        # ff = np.zeros((filter.shape[0], filter.shape[0]))
        # ff[:, 0] = filter
        # filter = ff
        for rejection in rejections:
            x = np.dot(x, rejection)
