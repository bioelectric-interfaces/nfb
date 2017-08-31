from PyQt4 import QtGui, QtCore

from pynfb.io.defaults import vectors_defaults as defaults
from pynfb.settings_widget import FileSelectorLine

default_signal = defaults['vSignals']['DerivedSignal'][0]
roi_labels = ['CUSTOM', 'bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh', 'caudalanteriorcingulate-rh',
              'caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh', 'cuneus-lh', 'cuneus-rh', 'entorhinal-lh',
              'entorhinal-rh', 'frontalpole-lh', 'frontalpole-rh', 'fusiform-lh', 'fusiform-rh', 'inferiorparietal-lh',
              'inferiorparietal-rh', 'inferiortemporal-lh', 'inferiortemporal-rh', 'insula-lh', 'insula-rh',
              'isthmuscingulate-lh', 'isthmuscingulate-rh', 'lateraloccipital-lh', 'lateraloccipital-rh',
              'lateralorbitofrontal-lh', 'lateralorbitofrontal-rh', 'lingual-lh', 'lingual-rh',
              'medialorbitofrontal-lh', 'medialorbitofrontal-rh', 'middletemporal-lh', 'middletemporal-rh',
              'paracentral-lh', 'paracentral-rh', 'parahippocampal-lh', 'parahippocampal-rh', 'parsopercularis-lh',
              'parsopercularis-rh', 'parsorbitalis-lh', 'parsorbitalis-rh', 'parstriangularis-lh',
              'parstriangularis-rh', 'pericalcarine-lh', 'pericalcarine-rh', 'postcentral-lh', 'postcentral-rh',
              'posteriorcingulate-lh', 'posteriorcingulate-rh', 'precentral-lh', 'precentral-rh', 'precuneus-lh',
              'precuneus-rh', 'rostralanteriorcingulate-lh', 'rostralanteriorcingulate-rh', 'rostralmiddlefrontal-lh',
              'rostralmiddlefrontal-rh', 'superiorfrontal-lh', 'superiorfrontal-rh', 'superiorparietal-lh',
              'superiorparietal-rh', 'superiortemporal-lh', 'superiortemporal-rh', 'supramarginal-lh',
              'supramarginal-rh', 'temporalpole-lh', 'temporalpole-rh', 'transversetemporal-lh',
              'transversetemporal-rh', 'unknown-lh']


class SpatialFilterROIWidget(QtGui.QWidget):
    def __init__(self):
        super(SpatialFilterROIWidget, self).__init__()
        layout = QtGui.QVBoxLayout(self)

        # labels
        self.labels = QtGui.QComboBox()
        for label in roi_labels:
            self.labels.addItem(label)
        layout.addWidget(self.labels)

        # spatial filter file
        self.file = FileSelectorLine(parent=self)
        layout.addWidget(self.file)

        # disable file selection if not CUSTOM
        self.labels.currentIndexChanged.connect(
            lambda: self.file.setEnabled(self.labels.currentIndex() == 0))

class SignalsSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vSignals']['DerivedSignal']

        # layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        # label
        label = QtGui.QLabel('Signals:')
        layout.addWidget(label)

        # list of signals
        self.list = QtGui.QListWidget(self)
        self.reset_items()
        self.list.itemDoubleClicked.connect(self.item_double_clicked_event)
        layout.addWidget(self.list)

        # buttons layout
        buttons_layout = QtGui.QHBoxLayout()
        add_button = QtGui.QPushButton('Add')
        add_button.clicked.connect(self.add)
        remove_signal_button = QtGui.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.remove_current_item)
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)

    def add(self):
        self.params.append(default_signal.copy())
        self.reset_items()
        self.signals_dialogs[-1].open()

    def remove_current_item(self, item):
        current = self.list.currentRow()
        if current >= 0:
            del self.params[current]
            self.reset_items()

    def item_double_clicked_event(self, item):
        self.signals_dialogs[self.list.currentRow()].open()

    def reset_items(self):
        self.params = self.parent().params['vSignals']['DerivedSignal']
        self.list.clear()
        self.signals_dialogs = []
        for signal in self.params:
            item = QtGui.QListWidgetItem(signal['sSignalName'])
            self.signals_dialogs.append(SignalDialog(self, signal_name=signal['sSignalName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setItemSelected(self.list.item(0), True)


class SignalDialog(QtGui.QDialog):
    def __init__(self, parent, signal_name='Signal'):
        self.params = parent.params
        super().__init__(parent)
        self.parent_list = parent
        self.setWindowTitle('Properties: ' + signal_name)
        self.form_layout = QtGui.QFormLayout(self)

        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(signal_name)
        self.form_layout.addRow('&Name:', self.name)
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[a-zA-Z0-9_]+$"))
        self.name.setValidator(validator)

        # spatial filter
        self.spatial_filter = SpatialFilterROIWidget()
        self.form_layout.addRow('Spatial filter:', self.spatial_filter)

        # disable spectrum evaluation
        self.disable_spectrum = QtGui.QCheckBox()
        self.disable_spectrum.stateChanged.connect(self.disable_spectrum_event)
        self.form_layout.addRow('&Disable spectrum \nevaluation:', self.disable_spectrum)

        # bandpass
        self.bandpass_low = QtGui.QSpinBox()
        self.bandpass_low.setRange(0, 250)
        self.bandpass_low.setValue(0)
        self.bandpass_high = QtGui.QSpinBox()
        self.bandpass_high.setRange(0, 250)
        self.bandpass_high.setValue(250)
        bandpass_widget = QtGui.QWidget()
        bandpass_layout = QtGui.QHBoxLayout(bandpass_widget)
        bandpass_layout.setMargin(0)
        label = QtGui.QLabel('low:')
        label.setMaximumWidth(20)
        bandpass_layout.addWidget(label)
        bandpass_layout.addWidget(self.bandpass_low)
        label = QtGui.QLabel('high:')
        label.setMaximumWidth(25)
        bandpass_layout.addWidget(label)
        bandpass_layout.addWidget(self.bandpass_high)
        self.form_layout.addRow('&Bandpass [Hz]:', bandpass_widget)

        # fft window size
        self.window_size = QtGui.QSpinBox()
        self.window_size.setRange(1, 100000)
        self.form_layout.addRow('&Window size:', self.window_size)

        # type
        self.type_list = QtGui.QComboBox()
        for protocol_type in ['fft', 'savgol', 'identity']:
            self.type_list.addItem(protocol_type)
        #self.form_layout.addRow('&Envelope detector type:', self.type_list)

        # exponential smoothing factor
        self.smoothing_factor = QtGui.QDoubleSpinBox()
        self.smoothing_factor.setRange(0, 1)
        self.smoothing_factor.setSingleStep(0.1)
        self.form_layout.addRow('&Smoothing factor:', self.smoothing_factor)

        # bci signal
        self.bci_checkbox = QtGui.QCheckBox('BCI mode')
        self.form_layout.addRow('&BCI mode:', self.bci_checkbox)

        self.temporal_settings = TemporalSettings()
        #self.form_layout.addRow('&Temporal settings:', self.temporal_settings)

        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def open(self):
        self.reset_items()
        super().open()

    def reset_items(self):
        current_signal_index = self.parent().list.currentRow()
        self.disable_spectrum.setChecked(self.params[current_signal_index]['bDisableSpectrumEvaluation'])
        self.bci_checkbox.setChecked(self.params[current_signal_index]['bBCIMode'])
        self.bandpass_low.setValue(self.params[current_signal_index]['fBandpassLowHz'])
        self.bandpass_high.setValue(self.params[current_signal_index]['fBandpassHighHz'])
        self.window_size.setValue(self.params[current_signal_index]['fFFTWindowSize'])
        self.smoothing_factor.setValue(self.params[current_signal_index]['fSmoothingFactor'])
        self.spatial_filter.file.path.setText(self.params[current_signal_index]['SpatialFilterMatrix'])
        roi_label = self.params[current_signal_index]['sROILabel']
        roi_label = 'CUSTOM' if roi_label == '' else roi_label
        self.spatial_filter.labels.setCurrentIndex(
            self.spatial_filter.labels.findText(roi_label, QtCore.Qt.MatchFixedString))
        self.temporal_settings.set_params(self.params[current_signal_index])

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sSignalName'] = self.name.text()
        self.params[current_signal_index]['fBandpassLowHz'] = self.bandpass_low.value()
        self.params[current_signal_index]['fBandpassHighHz'] = self.bandpass_high.value()
        self.params[current_signal_index]['SpatialFilterMatrix'] = self.spatial_filter.file.path.text()
        roi_labels = self.spatial_filter.labels.currentText()
        self.params[current_signal_index]['sROILabel'] = roi_labels if roi_labels != 'CUSTOM' else ''
        self.params[current_signal_index]['bDisableSpectrumEvaluation'] = int(self.disable_spectrum.isChecked())
        self.params[current_signal_index]['bBCIMode'] = int(self.bci_checkbox.isChecked())
        self.params[current_signal_index]['fFFTWindowSize'] = self.window_size.value()
        self.params[current_signal_index]['fSmoothingFactor'] = self.smoothing_factor.value()
        for key, val in self.temporal_settings.get_params().items():
            self.params[current_signal_index][key] = val
        self.parent().reset_items()
        self.close()

    def disable_spectrum_event(self):
        flag = self.disable_spectrum.isChecked()
        self.bandpass_low.setDisabled(flag)
        self.bandpass_high.setDisabled(flag)
        self.window_size.setDisabled(flag)
        self.smoothing_factor.setDisabled(flag)


class BandWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.low = QtGui.QSpinBox()
        self.low.setRange(0, 250)
        self.low.setValue(0)
        self.high = QtGui.QSpinBox()
        self.high.setRange(0, 250)
        self.high.setValue(250)
        bandpass_layout = QtGui.QHBoxLayout(self)
        bandpass_layout.setMargin(0)
        label = QtGui.QLabel('low:')
        label.setMaximumWidth(20)
        bandpass_layout.addWidget(label)
        bandpass_layout.addWidget(self.low)
        label = QtGui.QLabel('high:')
        label.setMaximumWidth(25)
        bandpass_layout.addWidget(label)
        bandpass_layout.addWidget(self.high)


class TemporalSettings(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # type
        self.type = QtGui.QComboBox()
        for protocol_type in ['envdetector', 'filter', 'identity']:
            self.type.addItem(protocol_type)


        # band
        self.band = BandWidget()

        # filter type
        self.filter_type = QtGui.QComboBox()
        for protocol_type in ['fft', 'butter', 'complexdem']:
            self.filter_type.addItem(protocol_type)

        # filter order
        self.order = QtGui.QSpinBox()
        self.order.setRange(1, 7)
        self.order.setValue(2)

        # filter order
        self.win_size = QtGui.QSpinBox()
        self.win_size.setRange(2, 5000)
        self.win_size.setValue(500)

        # smoother type
        self.smoother_type = QtGui.QComboBox()
        for protocol_type in ['exp', 'savgol']:
            self.smoother_type.addItem(protocol_type)

        # filter order
        self.smoother_factor = QtGui.QDoubleSpinBox()
        self.smoother_factor.setRange(0, 1)
        self.smoother_factor.setValue(0.3)

        layout = QtGui.QFormLayout(self)
        layout.addRow('&Type:', self.type)
        layout.addRow('&Band:', self.band)
        layout.addRow('&Filter type:', self.filter_type)
        layout.addRow('&Window size:', self.win_size)
        layout.addRow('&Filter order:', self.order)
        layout.addRow('&Smoother type:', self.smoother_type)
        layout.addRow('&Smoother factor:', self.smoother_factor)

        # setup disable
        self.smoother_type.currentIndexChanged.connect(self.smoother_type_changed)
        self.filter_type.currentIndexChanged.connect(self.filter_type_changed)
        self.type.currentIndexChanged.connect(self.type_changed)

        #self.type.currentIndexChanged.connect(lambda: self.set_disable_condition(
        #    self.type.currentText() == 'Identity',
         #   [self.order, self.win_size, self.band, self.smoother_factor, self.smoother_type, self.filter_type]))


    def type_changed(self):
        if self.type.currentText() == 'identity':
            for w in [self.order, self.win_size, self.band, self.smoother_factor, self.smoother_type, self.filter_type]:
                w.setDisabled(True)
        elif self.type.currentText() == 'filter':
            for w in [self.win_size, self.order, self.smoother_factor, self.smoother_type, self.filter_type]:
                w.setDisabled(True)
            for w in [self.band]:
                w.setEnabled(True)
        else:
            for w in [self.order, self.win_size, self.band, self.smoother_factor, self.smoother_type, self.filter_type]:
                w.setEnabled(True)
        self.filter_type_changed()
        self.smoother_type_changed()

    def filter_type_changed(self):
        if self.filter_type.isEnabled():
            self.win_size.setEnabled(self.filter_type.currentText() == 'fft')
            self.order.setEnabled(self.filter_type.currentText() != 'fft')

    def smoother_type_changed(self):
        if self.smoother_type.isEnabled():
            self.smoother_factor.setEnabled(self.filter_type.currentText() == 'exp')

    def set_disable_condition(self, condition, widgets):
        for widget in widgets:
            widget.setDisabled(condition)

    def set_params(self, dict):
        self.type.setCurrentIndex(self.type.findText(dict['sTemporalType'], QtCore.Qt.MatchFixedString))
        self.band.low.setValue(dict['fBandpassLowHz'])
        self.band.high.setValue(dict['fBandpassHighHz'])
        self.filter_type.setCurrentIndex(self.filter_type.findText(dict['sTemporalFilterType'], QtCore.Qt.MatchFixedString))
        self.win_size.setValue(dict['fFFTWindowSize'])
        self.order.setValue(dict['fTemporalFilterButterOrder'])
        self.smoother_type.setCurrentIndex(
            self.smoother_type.findText(dict['sTemporalSmootherType'], QtCore.Qt.MatchFixedString))
        self.smoother_factor.setValue(dict['fSmoothingFactor'])
        self.type_changed()
        self.filter_type_changed()
        self.smoother_type_changed()



    def get_params(self):
        params = dict()
        params['sTemporalType'] = self.type.currentText()
        params['fBandpassLowHz'] = self.band.low.value()
        params['fBandpassHighHz'] = self.band.high.value()
        params['sTemporalFilterType'] = self.filter_type.currentText()
        params['fFFTWindowSize'] = self.win_size.value()
        params['fTemporalFilterButterOrder'] = self.order.value()
        params['sTemporalSmootherType'] = self.smoother_type.currentText()
        params['fSmoothingFactor'] = self.smoother_factor.value()
        return params

if __name__ == '__main__':
    a = QtGui.QApplication([])
    w = TemporalSettings()
    w.show()
    a.exec_()