from PyQt4 import QtGui, QtCore

from pynfb.io.defaults import vectors_defaults as defaults
from pynfb.settings_widget import FileSelectorLine

default_signal = defaults['vSignals']['DerivedSignal'][0]


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
        self.spatial_filter = FileSelectorLine(parent=self)
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
        self.form_layout.addRow('&FFT window size:', self.window_size)

        # exponential smoothing factor
        self.smoothing_factor = QtGui.QDoubleSpinBox()
        self.smoothing_factor.setRange(0, 1)
        self.smoothing_factor.setSingleStep(0.1)
        self.form_layout.addRow('&Smoothing factor:', self.smoothing_factor)

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
        self.bandpass_low.setValue(self.params[current_signal_index]['fBandpassLowHz'])
        self.bandpass_high.setValue(self.params[current_signal_index]['fBandpassHighHz'])
        self.window_size.setValue(self.params[current_signal_index]['fFFTWindowSize'])
        self.smoothing_factor.setValue(self.params[current_signal_index]['fSmoothingFactor'])
        self.spatial_filter.path.setText(self.params[current_signal_index]['SpatialFilterMatrix'])

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sSignalName'] = self.name.text()
        self.params[current_signal_index]['fBandpassLowHz'] = self.bandpass_low.value()
        self.params[current_signal_index]['fBandpassHighHz'] = self.bandpass_high.value()
        self.params[current_signal_index]['SpatialFilterMatrix'] = self.spatial_filter.path.text()
        self.params[current_signal_index]['bDisableSpectrumEvaluation'] = int(self.disable_spectrum.isChecked())
        self.params[current_signal_index]['fFFTWindowSize'] = self.window_size.value()
        self.params[current_signal_index]['fSmoothingFactor'] = self.smoothing_factor.value()
        self.parent().reset_items()
        self.close()

    def disable_spectrum_event(self):
        flag = self.disable_spectrum.isChecked()
        self.bandpass_low.setDisabled(flag)
        self.bandpass_high.setDisabled(flag)
        self.window_size.setDisabled(flag)
        self.smoothing_factor.setDisabled(flag)