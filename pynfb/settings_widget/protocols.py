from PyQt4 import QtGui, QtCore

from pynfb.io.defaults import vectors_defaults as defaults

protocol_default = defaults['vProtocols']['FeedbackProtocol'][0]
protocols_types = ['Baseline', 'CircleFeedback', 'ThresholdBlink']


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
        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.path = QtGui.QLineEdit('')
        # self.path.textChanged.connect(self.raw_path_changed_event)
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

        # name line edit
        self.name = QtGui.QLineEdit(self)
        self.name.setText(protocol_name)
        self.form_layout.addRow('&Name:', self.name)

        # duration spin box
        self.duration = QtGui.QSpinBox()
        self.duration.setRange(0, 1000000)
        # self.duration.setValue(protocol_default['fDuration'])
        self.form_layout.addRow('&Duration [s]:', self.duration)

        # update statistics in the end end ssd analysis in the end check boxes
        self.update_statistics = QtGui.QCheckBox()
        self.ssd_in_the_end = QtGui.QCheckBox()
        self.ssd_in_the_end.clicked.connect(self.update_source_signal_combo_box)
        self.ssd_in_the_end.clicked.connect(lambda: self.update_statistics.setDisabled(self.ssd_in_the_end.isChecked()))
        self.form_layout.addRow('&SSD in the end:', self.ssd_in_the_end)
        self.form_layout.addRow('&Update statistics:', self.update_statistics)

        # source signal combo box
        self.source_signal = QtGui.QComboBox()
        self.form_layout.addRow('&Signal:', self.source_signal)
        # self.source_signal.currentIndexChanged.connect(self.source_signal_changed_event)

        # feedback type
        self.type = QtGui.QComboBox()
        for protocol_type in protocols_types:
            self.type.addItem(protocol_type)
        self.type.currentIndexChanged.connect(self.set_enabled_threshold_blink_settings)
        self.type.currentIndexChanged.connect(self.set_enabled_mock_settings)
        self.type.currentIndexChanged.connect(self.update_source_signal_combo_box)
        # self.type.setCurrentIndex(protocols_types.index(self.params))
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

        # mock settings
        # self.mock_checkbox = QtGui.QCheckBox()
        # self.form_layout.addRow('&Enable mock signals:', self.mock_checkbox)
        self.mock_file = FileSelectorLine()
        self.form_layout.addRow('&Mock signals file:', self.mock_file)
        self.mock_dataset = QtGui.QLineEdit('protocol1')
        self.form_layout.addRow('&Mock signals file\ndataset:', self.mock_dataset)
        self.set_enabled_mock_settings()

        # message text edit
        self.message = QtGui.QTextEdit()
        self.message.setMaximumHeight(50)
        self.form_layout.addRow('&Message:', self.message)

        # reward settings
        self.reward_signal = QtGui.QComboBox()
        self.update_reward_signal_combo_box()
        self.form_layout.addRow('&Reward signal:', self.reward_signal)
        self.reward_threshold = QtGui.QDoubleSpinBox()
        self.reward_threshold.setRange(-10000, 10000)
        self.form_layout.addRow('&Reward threshold:', self.reward_threshold)
        self.show_reward = QtGui.QCheckBox()
        self.form_layout.addRow('&Show reward:', self.show_reward)


        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def update_source_signal_combo_box(self):
        self.source_signal.clear()
        if self.type.currentText() == 'Baseline' and not self.ssd_in_the_end.isChecked():
            self.source_signal.addItem('All')
        for signal in self.parent().parent().params['vSignals']['DerivedSignal']:
            self.source_signal.addItem(signal['sSignalName'])

    def update_reward_signal_combo_box(self):
        for signal in self.parent().parent().params['vSignals']['DerivedSignal']:
            self.reward_signal.addItem(signal['sSignalName'])

    def set_enabled_threshold_blink_settings(self):
        flag = (self.type.currentText() == 'ThresholdBlink')
        self.blink_threshold.setEnabled(flag)
        self.blink_duration_ms.setEnabled(flag)

    def set_enabled_mock_settings(self):
        flag = (self.type.currentText() == 'CircleFeedback')
        self.mock_file.setEnabled(flag)
        self.mock_dataset.setEnabled(flag)

    def open(self):
        self.update_source_signal_combo_box()
        self.update_reward_signal_combo_box()
        self.reset_items()
        super().open()

    def reset_items(self):
        current_protocol = self.params[self.parent().list.currentRow()]
        print(current_protocol)
        self.duration.setValue(current_protocol['fDuration'])
        self.update_statistics.setChecked(current_protocol['bUpdateStatistics'])
        self.ssd_in_the_end.setChecked(current_protocol['bSSDInTheEnd'])
        self.update_statistics.setDisabled(self.ssd_in_the_end.isChecked())
        self.source_signal.setCurrentIndex(
            self.source_signal.findText(current_protocol['fbSource'], QtCore.Qt.MatchFixedString))
        self.type.setCurrentIndex(
            self.type.findText(current_protocol['sFb_type'], QtCore.Qt.MatchFixedString))
        self.blink_duration_ms.setValue(current_protocol['fBlinkDurationMs'])
        self.blink_threshold.setValue(current_protocol['fBlinkThreshold'])
        self.mock_file.path.setText(current_protocol['sMockSignalFilePath'])
        self.mock_dataset.setText(current_protocol['sMockSignalFileDataset'])
        self.message.setText(current_protocol['cString'])
        current_index = self.reward_signal.findText(current_protocol['sRewardSignal'], QtCore.Qt.MatchFixedString)
        self.reward_signal.setCurrentIndex(current_index if current_index>-1 else 0)
        self.show_reward.setChecked(current_protocol['bShowReward'])
        self.reward_threshold.setValue(current_protocol['bRewardThreshold'])
        pass

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sProtocolName'] = self.name.text()
        self.params[current_signal_index]['fDuration'] = self.duration.value()
        self.params[current_signal_index]['bUpdateStatistics'] = int(self.update_statistics.isChecked())
        self.params[current_signal_index]['bSSDInTheEnd'] = int(self.ssd_in_the_end.isChecked())
        self.params[current_signal_index]['fbSource'] = self.source_signal.currentText()
        self.params[current_signal_index]['sFb_type'] = self.type.currentText()
        self.params[current_signal_index]['fBlinkDurationMs'] = self.blink_duration_ms.value()
        self.params[current_signal_index]['fBlinkThreshold'] = self.blink_threshold.value()
        self.params[current_signal_index]['sMockSignalFilePath'] = self.mock_file.path.text()
        self.params[current_signal_index]['sMockSignalFileDataset'] = self.mock_dataset.text()
        self.params[current_signal_index]['cString'] = self.message.toPlainText()
        self.params[current_signal_index]['sRewardSignal'] = self.reward_signal.currentText()
        self.params[current_signal_index]['bShowReward'] = int(self.show_reward.isChecked())
        self.params[current_signal_index]['bRewardThreshold'] = self.reward_threshold.value()
        self.parent().reset_items()
        self.close()