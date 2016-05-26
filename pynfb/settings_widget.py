import sys
from PyQt4 import QtGui, QtCore
from collections import OrderedDict

from pynfb.experiment import Experiment

signals = [{'sSignalName': 'Signal1',
            'fBandpassLowHz': 1,
            'fBandpassHighHz': 10},
           {'sSignalName': 'Signal2',
            'fBandpassLowHz': 1,
            'fBandpassHighHz': 30}
           ]

default_signal = {'sSignalName': 'Unnamed Signal',
                  'fBandpassLowHz': 0,
                  'fBandpassHighHz': 250}

protocol_default = {'sProtocolName': 'Unnamed Protocol',
                    'bUpdateStatistics': 0,
                    'fDuration': 10,
                    'fbSource': 'All',
                    'sFb_type': 'Baseline'}

protocols_types = ['Baseline', 'Circle']

protocols = [{'sProtocolName': 'Circle feedback',
              'fDuration': 102,
              'bUpdateStatistics': 1,
              'fbSource': 'Signal2',
              'sFb_type': 'Circle'},
             {'sProtocolName': 'Baseline',
              'fDuration': 10,
              'bUpdateStatistics': True,
              'fbSource': 'All',
              'sFb_type': 'Baseline'}
             ]


protocols_sequence = ['Circle feedback']

parameters = {'vSignals': signals,
              'vProtocols': protocols,
              'vPSequence': protocols_sequence}

parameters_defaults = {'vSignals': [default_signal],
                       'vProtocols': [protocol_default],
                       'vPSequence': [],
                       'sExperimentName': 'experiment',
                       'sStreamName': 'NVX136_Data',
                       'sRawDataFilePath': ''}

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
        self.bandpass_low.setValue(self.params[self.parent().list.currentRow()]['fBandpassLowHz'])
        self.bandpass_high.setValue(self.params[self.parent().list.currentRow()]['fBandpassHighHz'])

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sSignalName'] = self.name.text()
        self.params[current_signal_index]['fBandpassLowHz'] = self.bandpass_low.value()
        self.params[current_signal_index]['fBandpassHighHz'] = self.bandpass_high.value()
        self.parent().reset_items()
        self.close()


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
        #self.type.setCurrentIndex(protocols_types.index(self.params))
        self.form_layout.addRow('&Type:', self.type)
        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def update_combo_box(self):
        self.source_signal.clear()
        for signal in self.parent().parent().params['vSignals']:
            self.source_signal.addItem(signal['sSignalName'])
        self.source_signal.addItem('All')

    def open(self):
        self.update_combo_box()
        self.reset_items()
        super().open()

    def reset_items(self):
        current_protocol = self.params[self.parent().list.currentRow()]
        self.duration.setValue(current_protocol['fDuration'])
        self.update_statistics.setChecked(current_protocol['bUpdateStatistics'])
        self.source_signal.setCurrentIndex(
            self.source_signal.findText(current_protocol['fbSource'], QtCore.Qt.MatchFixedString))
        self.type.setCurrentIndex(
            self.type.findText(current_protocol['sFb_type'], QtCore.Qt.MatchFixedString))
        pass

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sProtocolName'] = self.name.text()
        self.params[current_signal_index]['fDuration'] = self.duration.value()
        self.params[current_signal_index]['bUpdateStatistics'] = int(self.update_statistics.isChecked())
        self.params[current_signal_index]['fbSource'] = self.source_signal.currentText()
        self.params[current_signal_index]['sFb_type'] = self.type.currentText()
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
        self.combo.addItem('LSL stream name')
        self.combo.addItem('Raw file path')
        self.text = QtGui.QLineEdit()
        self.text.textChanged.connect(self.text_changed_event)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.combo)
        layout.addWidget(self.text)
        layout.setMargin(0)
        self.setLayout(layout)
        self.combo.currentIndexChanged.connect(self.combo_changed_event)
        self.combo_changed_event()

    def combo_changed_event(self):
        if self.combo.currentText()=='LSL stream name':
            self.text.setPlaceholderText('Print LSL stream name')
            self.text.setText(self.parent().params['sStreamName'])
        elif self.combo.currentText()=='Raw file path':
            self.text.setPlaceholderText('Print raw data file to stream')
            self.text.setText(self.parent().params['sRawDataFilePath'])

    def text_changed_event(self):
        if self.combo.currentText() == 'LSL stream name':
            self.parent().params['sStreamName'] = self.text.text()
        elif self.combo.currentText() == 'Raw file path':
            self.parent().params['sRawDataFilePath'] = self.text.text()
            print('change')

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
        self.form_layout.addRow('&Composite montage:', self.montage)
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

        self.params = parameters_defaults.copy()
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
        start_button.setIcon(QtGui.QIcon('static/imag/play-button.png'))
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
    window = SettingsWidget()
    window.show()
    sys.exit(app.exec_())
