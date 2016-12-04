from PyQt4 import QtGui, QtCore

from pynfb.io.defaults import vectors_defaults as defaults

protocol_default = defaults['vProtocols']['FeedbackProtocol'][0]
protocols_types = ['Baseline', 'CircleFeedback', 'ThresholdBlink', 'Video']


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
        self.form_layout.addRow('&Open signal manager\nin the end (SSD, CSP, ICA):', self.ssd_in_the_end)
        self.form_layout.addRow('&Update statistics:', self.update_statistics)

        # make a pause after protocol
        self.pause_after = QtGui.QCheckBox()
        self.form_layout.addRow('&Make a pause after protocol:', self.pause_after)

        # outliers
        self.drop_outliers = QtGui.QSpinBox()
        self.form_layout.addRow('&Drop outliers [std]:', self.drop_outliers)
        self.update_statistics.stateChanged.connect(lambda:
                                                    self.drop_outliers.setEnabled(self.update_statistics.isChecked()))

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
        self.type.currentIndexChanged.connect(
            lambda: self.circle_border.setEnabled(self.type.currentText() == 'CircleFeedback'))
        self.type.currentIndexChanged.connect(
            lambda: self.m_signal.setEnabled(self.type.currentText() == 'CircleFeedback'))
        self.type.currentIndexChanged.connect(
            lambda: self.m_signal_threshold.setEnabled(self.type.currentText() == 'CircleFeedback'))
        self.type.currentIndexChanged.connect(
            lambda: self.video_path.setEnabled(self.type.currentText() == 'Video'))
        # self.type.setCurrentIndex(protocols_types.index(self.params))
        self.form_layout.addRow('&Type:', self.type)

        # random circle bound
        self.circle_border = QtGui.QComboBox()
        self.circle_border.addItems(['Sin', 'Random'])
        self.form_layout.addRow('&Circle border:', self.circle_border)


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

        # mock previous
        mock_previos_layput = QtGui.QHBoxLayout()
        self.mock_previous = QtGui.QSpinBox()
        self.mock_previous.setRange(1, 100)
        self.enable_mock_previous = QtGui.QCheckBox()
        self.enable_mock_previous.stateChanged.connect(self.handle_enable_mock_previous)
        self.reverse_mock_previous = QtGui.QCheckBox('Reverse')
        mock_previos_layput.addWidget(self.enable_mock_previous)
        mock_previos_layput.addWidget(QtGui.QLabel('Protocol #'))
        mock_previos_layput.addWidget(self.mock_previous)
        mock_previos_layput.addWidget(self.reverse_mock_previous)
        self.form_layout.addRow('Mock from previous\nprotocol raw data', mock_previos_layput)

        # muscular signal
        self.m_signal = QtGui.QComboBox()
        self.m_signal_threshold = QtGui.QDoubleSpinBox()
        self.m_signal_threshold.setSingleStep(0.01)
        muscular_layout = QtGui.QHBoxLayout()
        muscular_layout.addWidget(self.m_signal)
        muscular_layout.addWidget(QtGui.QLabel('Threshold:'))
        muscular_layout.addWidget(self.m_signal_threshold)
        self.form_layout.addRow('Muscular signal:', muscular_layout)

        # message text edit
        self.message = QtGui.QTextEdit()
        self.message.setMaximumHeight(50)
        self.form_layout.addRow('&Message:', self.message)

        # split record (CSP)
        self.split_checkbox = QtGui.QCheckBox()
        self.form_layout.addRow('&Add half time\nextra message (for CSP):', self.split_checkbox)
        self.message2 = QtGui.QTextEdit()
        self.message2.setMaximumHeight(50)
        self.form_layout.addRow('&Half time extra message:', self.message2)
        self.split_checkbox.stateChanged.connect(lambda: self.message2.setEnabled(self.split_checkbox.isChecked()))
        self.message2.setEnabled(False)

        # reward settings
        self.reward_signal = QtGui.QComboBox()
        self.form_layout.addRow('&Reward signal:', self.reward_signal)
        self.reward_threshold = QtGui.QDoubleSpinBox()
        self.reward_threshold.setRange(-10000, 10000)
        self.form_layout.addRow('&Reward threshold:', self.reward_threshold)
        self.show_reward = QtGui.QCheckBox()
        self.form_layout.addRow('&Show reward:', self.show_reward)

        # video path
        self.video_path = FileSelectorLine()
        self.form_layout.addRow('&Video file:', self.video_path)

        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def handle_enable_mock_previous(self):
        self.mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.reverse_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.set_enabled_mock_settings()

    def update_source_signal_combo_box(self):
        text = self.source_signal.currentText()
        self.source_signal.clear()
        self.reward_signal.clear()
        if self.type.currentText() == 'Baseline':
            self.source_signal.addItem('All')
        all_signals = self.parent().parent().params['vSignals']
        signals = all_signals['DerivedSignal'].copy()
        if not self.ssd_in_the_end.isChecked():
            signals += all_signals['CompositeSignal'].copy()
        for signal in signals:
            self.source_signal.addItem(signal['sSignalName'])
            self.reward_signal.addItem(signal['sSignalName'])

        current_index = self.source_signal.findText(text, QtCore.Qt.MatchFixedString)
        self.source_signal.setCurrentIndex(current_index if current_index > -1 else 0)

    def set_enabled_threshold_blink_settings(self):
        flag = (self.type.currentText() == 'ThresholdBlink')
        self.blink_threshold.setEnabled(flag)
        self.blink_duration_ms.setEnabled(flag)

    def set_enabled_mock_settings(self):
        flag = (self.type.currentText() == 'CircleFeedback' and not self.enable_mock_previous.isChecked())
        self.mock_file.setEnabled(flag)
        self.mock_dataset.setEnabled(flag)

    def set_enabled_video_settings(self):
        flag = self.type.currentText() == 'Video'

    def open(self):
        self.update_source_signal_combo_box()
        self.reset_items()
        self.update_source_signal_combo_box()
        super().open()

    def reset_items(self):
        current_protocol = self.params[self.parent().list.currentRow()]
        self.duration.setValue(current_protocol['fDuration'])
        self.update_statistics.setChecked(current_protocol['bUpdateStatistics'])
        self.pause_after.setChecked(current_protocol['bPauseAfter'])
        self.drop_outliers.setValue(current_protocol['iDropOutliers'])
        self.drop_outliers.setEnabled(self.update_statistics.isChecked())
        self.ssd_in_the_end.setChecked(current_protocol['bSSDInTheEnd'])
        self.source_signal.setCurrentIndex(
            self.source_signal.findText(current_protocol['fbSource'], QtCore.Qt.MatchFixedString))
        self.type.setCurrentIndex(
            self.type.findText(current_protocol['sFb_type'], QtCore.Qt.MatchFixedString))
        self.circle_border.setCurrentIndex(current_protocol['iRandomBound'])
        self.circle_border.setEnabled(self.type.currentText() == 'CircleFeedback')
        self.blink_duration_ms.setValue(current_protocol['fBlinkDurationMs'])
        self.blink_threshold.setValue(current_protocol['fBlinkThreshold'])
        self.mock_file.path.setText(current_protocol['sMockSignalFilePath'])
        self.mock_dataset.setText(current_protocol['sMockSignalFileDataset'])
        self.message.setText(current_protocol['cString'])
        self.message2.setText(current_protocol['cString2'])
        self.split_checkbox.setChecked(current_protocol['bUseExtraMessage'])
        current_index = self.reward_signal.findText(current_protocol['sRewardSignal'], QtCore.Qt.MatchFixedString)
        self.reward_signal.setCurrentIndex(current_index if current_index > -1 else 0)
        self.show_reward.setChecked(current_protocol['bShowReward'])
        self.reward_threshold.setValue(current_protocol['bRewardThreshold'])
        self.enable_mock_previous.setChecked(current_protocol['iMockPrevious'] > 0)
        self.mock_previous.setValue(current_protocol['iMockPrevious'])
        self.reverse_mock_previous.setChecked(current_protocol['bReverseMockPrevious'])
        self.mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.reverse_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.video_path.path.setText(current_protocol['sVideoPath'])
        self.video_path.setEnabled(self.type.currentText() == 'Video')
        signals = ([d['sSignalName'] for d in self.parent().parent().params['vSignals']['DerivedSignal']] +
                   [d['sSignalName'] for d in self.parent().parent().params['vSignals']['CompositeSignal']])
        self.m_signal.addItems(['None'] + signals)
        self.m_signal.setEnabled(self.type.currentText() == 'CircleFeedback')
        self.m_signal_threshold.setEnabled(self.type.currentText() == 'CircleFeedback')
        current_index = self.m_signal.findText(current_protocol['sMSignal'], QtCore.Qt.MatchFixedString)
        self.m_signal.setCurrentIndex(current_index if current_index > -1 else 0)
        self.m_signal_threshold.setValue(current_protocol['fMSignalThreshold'])
        pass

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sProtocolName'] = self.name.text()
        self.params[current_signal_index]['fDuration'] = self.duration.value()
        self.params[current_signal_index]['bUpdateStatistics'] = int(self.update_statistics.isChecked())
        self.params[current_signal_index]['bPauseAfter'] = int(self.pause_after.isChecked())
        self.params[current_signal_index]['iDropOutliers'] = (
            self.drop_outliers.value() if self.update_statistics.isChecked() else 0)
        self.params[current_signal_index]['bSSDInTheEnd'] = int(self.ssd_in_the_end.isChecked())
        self.params[current_signal_index]['fbSource'] = self.source_signal.currentText()
        self.params[current_signal_index]['sFb_type'] = self.type.currentText()
        self.params[current_signal_index]['iRandomBound'] = self.circle_border.currentIndex()
        self.params[current_signal_index]['fBlinkDurationMs'] = self.blink_duration_ms.value()
        self.params[current_signal_index]['fBlinkThreshold'] = self.blink_threshold.value()
        self.params[current_signal_index]['sMockSignalFilePath'] = self.mock_file.path.text()
        self.params[current_signal_index]['sMockSignalFileDataset'] = self.mock_dataset.text()
        self.params[current_signal_index]['cString'] = self.message.toPlainText()
        self.params[current_signal_index]['cString2'] = self.message2.toPlainText()
        self.params[current_signal_index]['bUseExtraMessage'] = int(self.split_checkbox.isChecked())
        self.params[current_signal_index]['sRewardSignal'] = self.reward_signal.currentText()
        self.params[current_signal_index]['bShowReward'] = int(self.show_reward.isChecked())
        self.params[current_signal_index]['bRewardThreshold'] = self.reward_threshold.value()
        self.params[current_signal_index]['iMockPrevious'] = (
            self.mock_previous.value() if self.enable_mock_previous.isChecked() else 0)
        self.params[current_signal_index]['bReverseMockPrevious'] = (
            int(self.reverse_mock_previous.isChecked()) if self.enable_mock_previous.isChecked() else 0)
        self.params[current_signal_index]['sVideoPath'] = self.video_path.path.text()
        self.params[current_signal_index]['sMSignal'] = self.m_signal.currentText()
        self.params[current_signal_index]['fMSignalThreshold'] = self.m_signal_threshold.value()
        self.parent().reset_items()
        self.close()