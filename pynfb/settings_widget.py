import sys
from PyQt4 import QtGui, QtCore
from pynfb.io.xml import xml_file_to_params, format_odict_by_defaults
from pynfb.io.defaults import vectors_defaults as defaults
from pynfb.experiment import Experiment
import os


static_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__))+'/static')

default_signal = defaults['vSignals']['DerivedSignal'][0]

protocol_default = defaults['vProtocols']['FeedbackProtocol'][0]

protocols_types = ['Baseline', 'Circle', 'ThresholdBlink']


inlet_types = ['lsl', 'lsl_from_file', 'lsl_generator', 'ftbuffer']


class SignalsSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vSignals']

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
        self.params = self.parent().params['vSignals']
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
        self.reset_items()
        super().open()

    def reset_items(self):
        current_signal_index = self.parent().list.currentRow()
        self.disable_spectrum.setChecked(self.params[current_signal_index]['bDisableSpectrumEvaluation'])
        self.bandpass_low.setValue(self.params[current_signal_index]['fBandpassLowHz'])
        self.bandpass_high.setValue(self.params[current_signal_index]['fBandpassHighHz'])
        self.spatial_filter.path.setText(self.params[current_signal_index]['SpatialFilterMatrix'])

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sSignalName'] = self.name.text()
        self.params[current_signal_index]['fBandpassLowHz'] = self.bandpass_low.value()
        self.params[current_signal_index]['fBandpassHighHz'] = self.bandpass_high.value()
        self.params[current_signal_index]['SpatialFilterMatrix'] = self.spatial_filter.path.text()
        self.params[current_signal_index]['bDisableSpectrumEvaluation'] = int(self.disable_spectrum.isChecked())
        self.parent().reset_items()
        self.close()

    def disable_spectrum_event(self):
        if self.disable_spectrum.isChecked():
            self.bandpass_low.setDisabled(True)
            self.bandpass_high.setDisabled(True)
        else:
            self.bandpass_low.setDisabled(False)
            self.bandpass_high.setDisabled(False)


class ProtocolsSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vProtocols']
        protocols_label = QtGui.QLabel('Protocols:')
        self.list = QtGui.QListWidget(self)

        self.list.setDragDropMode(QtGui.QAbstractItemView.DragOnly)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(protocols_label)
        layout.addWidget(self.list)
        self.reset_items()
        self.list.itemDoubleClicked.connect(self.item_double_clicked_event)
        buttons_layout = QtGui.QHBoxLayout()
        add_signal_button = QtGui.QPushButton('Add')
        add_signal_button.clicked.connect(self.add_action)
        remove_signal_button = QtGui.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.remove_current_item)
        buttons_layout.addWidget(add_signal_button)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        # self.show()

    def add_action(self):
        self.params.append(protocol_default.copy())
        self.reset_items()
        self.dialogs[-1].open()

    def remove_current_item(self, item):
        current = self.list.currentRow()
        if current >= 0:
            del self.params[current]
            self.reset_items()
        # self.show()

    def item_double_clicked_event(self, item):
        self.dialogs[self.list.currentRow()].open()

    def reset_items(self):
        self.params = self.parent().params['vProtocols']
        self.list.clear()
        self.dialogs = []
        for signal in self.params:
            item = QtGui.QListWidgetItem(signal['sProtocolName'])
            self.dialogs.append(ProtocolDialog(self, protocol_name=signal['sProtocolName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setItemSelected(self.list.item(0), True)


class FileSelectorLine(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        self.setContentsMargins(0,0,0,0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.path = QtGui.QLineEdit('')
        #self.path.textChanged.connect(self.raw_path_changed_event)
        self.select_button = QtGui.QPushButton('Select file...')
        self.select_button.clicked.connect(self.chose_file_action)
        layout.addWidget(self.path)
        layout.addWidget(self.select_button)

    def chose_file_action(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        self.path.setText(fname)

class ProtocolDialog(QtGui.QDialog):
    def __init__(self, parent, protocol_name='Protocol'):
        super().__init__(parent)
        self.params = parent.params
        self.parent_list = parent
        self.setWindowTitle('Properties: ' + protocol_name)
        self.form_layout = QtGui.QFormLayout(self)
        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(protocol_name)
        self.form_layout.addRow('&Name:', self.name)
        # duration
        self.duration =  QtGui.QSpinBox()
        self.duration.setRange(0, 1000000)
        #self.duration.setValue(protocol_default['fDuration'])
        self.form_layout.addRow('&Duration [s]:', self.duration)
        # update statistics in the end
        self.update_statistics = QtGui.QCheckBox()
        #self.update_statistics.setTristate(protocol_default['bUpdateStatistics'])
        self.form_layout.addRow('&Update statistics:', self.update_statistics)
        # source signal
        self.source_signal = QtGui.QComboBox()
        self.update_combo_box()
        self.form_layout.addRow('&Source signal:', self.source_signal)
        # feedback type
        self.type = QtGui.QComboBox()
        for protocol_type in protocols_types:
            self.type.addItem(protocol_type)
        self.type.currentIndexChanged.connect(self.set_enabled_threshold_blink_settings)
        #self.type.setCurrentIndex(protocols_types.index(self.params))
        self.form_layout.addRow('&Type:', self.type)
        # threshold blink settings
        self.blink_duration_ms = QtGui.QSpinBox()
        self.blink_duration_ms.setRange(0, 1000000)
        self.blink_threshold = QtGui.QDoubleSpinBox()
        self.blink_threshold.setRange(-1e20, 1e20)
        self.blink_threshold.setEnabled(False)
        self.blink_duration_ms.setEnabled(False)
        self.form_layout.addRow('&Blink duration [ms]:', self.blink_duration_ms)
        self.form_layout.addRow('&Blink threshold:', self.blink_threshold)
        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def update_combo_box(self):
        self.source_signal.clear()
        for signal in self.parent().parent().params['vSignals']:
            self.source_signal.addItem(signal['sSignalName'])
        self.source_signal.addItem('All')

    def set_enabled_threshold_blink_settings(self):
        flag = (self.type.currentText() == 'ThresholdBlink')
        self.blink_threshold.setEnabled(flag)
        self.blink_duration_ms.setEnabled(flag)



    def open(self):
        self.update_combo_box()
        self.reset_items()
        super().open()

    def reset_items(self):
        current_protocol = self.params[self.parent().list.currentRow()]
        print(current_protocol)
        self.duration.setValue(current_protocol['fDuration'])
        self.update_statistics.setChecked(current_protocol['bUpdateStatistics'])
        self.source_signal.setCurrentIndex(
            self.source_signal.findText(current_protocol['fbSource'], QtCore.Qt.MatchFixedString))
        self.type.setCurrentIndex(
            self.type.findText(current_protocol['sFb_type'], QtCore.Qt.MatchFixedString))
        self.blink_duration_ms.setValue(current_protocol['fBlinkDurationMs'])
        self.blink_threshold.setValue(current_protocol['fBlinkThreshold'])
        pass

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sProtocolName'] = self.name.text()
        self.params[current_signal_index]['fDuration'] = self.duration.value()
        self.params[current_signal_index]['bUpdateStatistics'] = int(self.update_statistics.isChecked())
        self.params[current_signal_index]['fbSource'] = self.source_signal.currentText()
        self.params[current_signal_index]['sFb_type'] = self.type.currentText()
        self.params[current_signal_index]['fBlinkDurationMs'] = self.blink_duration_ms.value()
        self.params[current_signal_index]['fBlinkThreshold'] = self.blink_threshold.value()
        self.parent().reset_items()
        self.close()


class ProtocolSequenceSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vPSequence']
        label = QtGui.QLabel('Protocols sequence:')
        self.list = ProtocolSequenceListWidget(parent=self)
        #self.list.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        #self.list.connect.
        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        buttons_layout = QtGui.QHBoxLayout()
        remove_signal_button = QtGui.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.list.remove_current_row)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def reset_items(self):
        self.params = self.parent().params['vPSequence']
        self.list.reset_items()




class ProtocolSequenceListWidget(QtGui.QListWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params
        self.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.reset_items()

    def dropEvent(self, QDropEvent):
        super().dropEvent(QDropEvent)
        self.save()


    def reset_items(self):
        self.params = self.parent().params

        self.clear()
        for protocol in self.params:
            item = QtGui.QListWidgetItem(protocol)
            self.addItem(item)

    def save(self):
        self.params = [self.item(j).text() for j in range(self.count())]

        self.parent().params = self.params
        self.parent().parent().params['vPSequence'] = self.params

    def remove_current_row(self):
        current = self.currentRow()
        if current >= 0:
            del self.params[current]
            self.reset_items()


class InletSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setContentsMargins(0,0,0,0)
        self.combo = QtGui.QComboBox()
        self.combo.addItem('LSL stream')
        self.combo.addItem('LSL from file')
        self.combo.addItem('LSL generator')
        self.combo.addItem('Field Trip buffer')
        self.line_edit_1 = QtGui.QLineEdit()
        self.line_edit_1.textChanged.connect(self.line_edit_1_changed_event)
        self.line_edit_2 = QtGui.QLineEdit()
        self.line_edit_2.textChanged.connect(self.line_edit_2_changed_event)
        #self.stream_name = QtGui.QLineEdit()
        #self.stream_name.textChanged.connect(self.stream_name_changed_event)
        #self.raw_path = QtGui.QLineEdit('')
        #self.raw_path.textChanged.connect(self.raw_path_changed_event)
        self.raw_select_button = QtGui.QPushButton('Select file...')
        self.raw_select_button.clicked.connect(self.chose_file_action)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.combo)
        layout.addWidget(self.line_edit_1)
        layout.addWidget(self.line_edit_2)
        #layout.addWidget(self.stream_name)
        #layout.addWidget(self.raw_path)
        layout.addWidget(self.raw_select_button)
        layout.setMargin(0)
        self.setLayout(layout)
        self.combo.currentIndexChanged.connect(self.combo_changed_event)
        self.combo.setCurrentIndex(inlet_types.index(self.parent().params['sInletType']))
        self.combo_changed_event()


    def line_edit_2_changed_event(self):
        host, port = self.parent().params['sFTHostnamePort'].split(':')
        self.parent().params['sFTHostnamePort'] = host + ':' + self.line_edit_2.text()

    def line_edit_1_changed_event(self):
        if self.combo.currentIndex() == 0:
            self.parent().params['sStreamName'] = self.line_edit_1.text()
        elif self.combo.currentIndex() == 1:
            self.parent().params['sRawDataFilePath'] = self.line_edit_1.text()
        elif self.combo.currentIndex() == 2:
            pass
        elif self.combo.currentIndex() == 3:
            host, port = self.parent().params['sFTHostnamePort'].split(':')
            self.parent().params['sFTHostnamePort'] = self.line_edit_1.text()+':'+port

    def combo_changed_event(self):
        self.parent().params['sInletType'] = inlet_types[self.combo.currentIndex()]
        self.raw_select_button.hide()
        self.line_edit_1.setEnabled(True)
        self.line_edit_2.hide()
        if self.combo.currentIndex()==0:
            self.line_edit_1.setPlaceholderText('Print LSL stream name')
            self.line_edit_1.setText(self.parent().params['sStreamName'])
        elif self.combo.currentIndex()==1:
            self.raw_select_button.show()
            self.line_edit_1.setPlaceholderText('Print raw data file path')
            self.line_edit_1.setText(self.parent().params['sRawDataFilePath'])
        elif self.combo.currentIndex()==2:
            self.line_edit_1.setPlaceholderText('')
            self.line_edit_1.setEnabled(False)
            self.line_edit_1.setText('')
        elif self.combo.currentIndex()==3:
            host, port = self.parent().params['sFTHostnamePort'].split(':')
            self.line_edit_2.show()
            self.line_edit_1.setPlaceholderText('Hostname')
            self.line_edit_2.setPlaceholderText('Port')
            self.line_edit_1.setText(host)
            self.line_edit_2.setText(port)

    def chose_file_action(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        self.line_edit_1.setText(fname)
        self.parent().params['sRawDataFilePath'] = fname

class GeneralSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params
        self.form_layout = QtGui.QFormLayout(self)
        self.setLayout(self.form_layout)
        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(self.params['sExperimentName'])
        self.name.textChanged.connect(self.name_changed_event)
        self.form_layout.addRow('&Name:', self.name)
        # composite montage
        self.montage = QtGui.QLineEdit(self)
        self.montage.setPlaceholderText('Print path to file')
        self.form_layout.addRow('&Composite\nmontage:', self.montage)
        # inlet
        self.inlet = InletSettingsWidget(parent=self)
        self.form_layout.addRow('&Inlet:', self.inlet)
        #self.stream

    def name_changed_event(self):
        self.params['sExperimentName'] = self.name.text()

    def reset(self):
        self.params = self.parent().params
        self.name.setText(self.params['sExperimentName'])
        self.inlet.combo_changed_event()


class SettingsWidget(QtGui.QWidget):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        v_layout = QtGui.QVBoxLayout()
        layout = QtGui.QHBoxLayout()
        self.params = xml_file_to_params()
        self.general_settings = GeneralSettingsWidget(parent=self)
        v_layout.addWidget(self.general_settings)
        v_layout.addLayout(layout)
        self.protocols_list = ProtocolsSettingsWidget(parent=self)
        self.signals_list = SignalsSettingsWidget(parent=self)
        self.protocols_sequence_list = ProtocolSequenceSettingsWidget(parent=self)
        #layout.addWidget(self.general_settings)
        layout.addWidget(self.signals_list)
        layout.addWidget(self.protocols_list)
        layout.addWidget(self.protocols_sequence_list)
        start_button = QtGui.QPushButton('Start')
        start_button.setIcon(QtGui.QIcon(static_path+'/imag/power-button.png'))
        start_button.setMinimumHeight(50)
        start_button.setMinimumWidth(200)
        start_button.clicked.connect(self.onClicked)
        name_layout = QtGui.QHBoxLayout()
        v_layout.addWidget(start_button, alignment = QtCore.Qt.AlignCenter)
        self.setLayout(v_layout)

    def reset_parameters(self):
        self.signals_list.reset_items()
        self.protocols_list.reset_items()
        self.protocols_sequence_list.reset_items()
        self.general_settings.reset()
        #self.params['sExperimentName'] = self.experiment_name.text()

    def onClicked(self):
        self.experiment = Experiment(self.app, self.params)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = FileSelectorLine()
    window.show()
    sys.exit(app.exec_())
