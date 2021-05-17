from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from pynfb.serializers.defaults import vectors_defaults as defaults
from pynfb.widgets.helpers import ScrollArea

protocol_default = defaults['vProtocols']['FeedbackProtocol'][0]
protocols_types = ['Baseline', 'Feedback', 'ThresholdBlink', 'Video']


class ProtocolsSettingsWidget(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vProtocols']
        protocols_label = QtWidgets.QLabel('Protocols:')
        self.list = QtWidgets.QListWidget(self)

        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(protocols_label)
        layout.addWidget(self.list)
        self.reset_items()
        self.list.itemDoubleClicked.connect(self.item_double_clicked_event)
        buttons_layout = QtWidgets.QHBoxLayout()
        add_signal_button = QtWidgets.QPushButton('Add')
        add_signal_button.clicked.connect(self.add_action)
        remove_signal_button = QtWidgets.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.remove_current_item)
        buttons_layout.addWidget(add_signal_button)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        # self.show()

    def add_action(self):
        self.params.append(protocol_default.copy())
        self.reset_items()
        self.list.setCurrentItem(self.list.item(len(self.params) - 1))
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
            item = QtWidgets.QListWidgetItem(signal['sProtocolName'])
            self.dialogs.append(ProtocolDialog(self, protocol_name=signal['sProtocolName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setCurrentItem(self.list.item(0))


class FileSelectorLine(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.path = QtWidgets.QLineEdit('')
        # self.path.textChanged.connect(self.raw_path_changed_event)
        self.select_button = QtWidgets.QPushButton('Select file...')
        self.select_button.clicked.connect(self.chose_file_action)
        layout.addWidget(self.path)
        layout.addWidget(self.select_button)

    def chose_file_action(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', './')[0]
        self.path.setText(fname)


class ProtocolDialog(QtWidgets.QDialog):
    def __init__(self, parent, protocol_name='Protocol'):
        super().__init__(parent)
        self.params = parent.params
        self.parent_list = parent
        self.setWindowTitle('Properties: ' + protocol_name)

        # Set up a scroll area
        dialogLayout = QtWidgets.QVBoxLayout(self)
        dialogLayout.setContentsMargins(0, 0, 0, 0)

        scrollArea = ScrollArea()
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollArea.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        scrollArea.setWidgetResizable(True)
        scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        dialogLayout.addWidget(scrollArea)

        scrollWidget = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QFormLayout(scrollWidget)
        scrollWidget.sizeHint = lambda: scrollWidget.layout().sizeHint()
        scrollWidget.setContentsMargins(0, 0, 0, 0)
        scrollArea.setWidget(scrollWidget)

        # name line edit
        self.name = QtWidgets.QLineEdit(self)
        self.name.setText(protocol_name)
        self.form_layout.addRow('&Name:', self.name)

        # duration spin box
        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(0.1, 1000000)
        # self.duration.setValue(protocol_default['fDuration'])
        self.form_layout.addRow('&Duration [s]:', self.duration)

        # duration spin box
        self.random_over_time = QtWidgets.QDoubleSpinBox()
        self.random_over_time.setRange(0, 1000000)
        # self.duration.setValue(protocol_default['fDuration'])
        self.form_layout.addRow('&Random over time [s]:', self.random_over_time)

        # update statistics in the end end ssd analysis in the end check boxes
        self.ssd_in_the_end = QtWidgets.QCheckBox()
        self.ssd_in_the_end.clicked.connect(self.update_source_signal_combo_box)
        self.form_layout.addRow('&Open signal manager\nin the end (SSD, CSP, ICA):', self.ssd_in_the_end)


        self.update_statistics = QtWidgets.QCheckBox()
        self.update_statistics_type = QtWidgets.QComboBox()
        self.update_statistics.stateChanged.connect(
            lambda: self.update_statistics_type.setEnabled(self.update_statistics.isChecked()))
        for s_type in ['meanstd', 'max']:
            self.update_statistics_type.addItem(s_type)

        stats_widget = QtWidgets.QWidget()
        stats_layout = QtWidgets.QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.addWidget(self.update_statistics)
        stats_layout.addWidget(self.update_statistics_type)
        self.form_layout.addRow('&Update statistics:', stats_widget)

        # make signal after protocol
        self.beep_after = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Beep after protocol:', self.beep_after)

        # fast bci fitting
        self.auto_bci_fit = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Auto BCI fitting:', self.auto_bci_fit)

        # make signal after protocol
        self.mock_source = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Mock source:', self.mock_source)

        # make a pause after protocol
        self.pause_after = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Make a pause after protocol:', self.pause_after)

        # enable detection task
        self.detection_task = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Enable detection task:', self.detection_task)

        # outliers
        self.drop_outliers = QtWidgets.QSpinBox()
        self.form_layout.addRow('&Drop outliers [std]:', self.drop_outliers)
        self.update_statistics.stateChanged.connect(lambda:
                                                    self.drop_outliers.setEnabled(self.update_statistics.isChecked()))

        # source signal combo box
        self.source_signal = QtWidgets.QComboBox()
        self.form_layout.addRow('&Signal:', self.source_signal)
        # self.source_signal.currentIndexChanged.connect(self.source_signal_changed_event)

        # feedback type
        self.type = QtWidgets.QComboBox()
        for protocol_type in protocols_types:
            self.type.addItem(protocol_type)
        self.type.currentIndexChanged.connect(self.set_enabled_threshold_blink_settings)
        self.type.currentIndexChanged.connect(self.set_enabled_mock_settings)
        self.type.currentIndexChanged.connect(self.update_source_signal_combo_box)
        self.type.currentIndexChanged.connect(
            lambda: self.circle_border.setEnabled(self.type.currentText() == 'Feedback'))
        self.type.currentIndexChanged.connect(
            lambda: self.m_signal.setEnabled(self.type.currentText() == 'Feedback'))
        self.type.currentIndexChanged.connect(
            lambda: self.m_signal_threshold.setEnabled(self.type.currentText() == 'Feedback'))
        self.type.currentIndexChanged.connect(
            lambda: self.video_path.setEnabled(self.type.currentText() == 'Video'))
        # self.type.setCurrentIndex(protocols_types.index(self.params))
        self.form_layout.addRow('&Type:', self.type)

        # random circle bound
        self.circle_border = QtWidgets.QComboBox()
        self.circle_border.addItems(['SinCircle', 'RandomCircle', 'Bar'])
        self.form_layout.addRow('&Feedback type:', self.circle_border)


        # threshold blink settings
        self.blink_duration_ms = QtWidgets.QSpinBox()
        self.blink_duration_ms.setRange(0, 1000000)
        self.blink_threshold = QtWidgets.QDoubleSpinBox()
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
        self.mock_dataset = QtWidgets.QLineEdit('protocol1')
        self.form_layout.addRow('&Mock signals file\ndataset:', self.mock_dataset)


        # mock previous
        mock_previos_layput = QtWidgets.QHBoxLayout()
        self.mock_previous = QtWidgets.QSpinBox()
        self.mock_previous.setRange(1, 100)
        self.enable_mock_previous = QtWidgets.QCheckBox()
        self.enable_mock_previous.stateChanged.connect(self.handle_enable_mock_previous)
        self.reverse_mock_previous = QtWidgets.QCheckBox('Reverse')
        self.random_mock_previous = QtWidgets.QCheckBox('Shuffle')
        # self.random_mock_previous.hide()
        self.random_mock_previous.stateChanged.connect(self.handle_random_mock_previous)
        mock_previos_layput.addWidget(self.enable_mock_previous)
        mock_previos_layput.addWidget(QtWidgets.QLabel('Protocol #'))
        mock_previos_layput.addWidget(self.mock_previous)
        mock_previos_layput.addWidget(self.random_mock_previous)
        mock_previos_layput.addWidget(self.reverse_mock_previous)
        self.form_layout.addRow('Mock from previous\nprotocol raw data', mock_previos_layput)

        # enable mock
        self.set_enabled_mock_settings()

        # muscular signal
        self.m_signal = QtWidgets.QComboBox()
        self.m_signal_threshold = QtWidgets.QDoubleSpinBox()
        self.m_signal_threshold.setSingleStep(0.01)
        muscular_layout = QtWidgets.QHBoxLayout()
        muscular_layout.addWidget(self.m_signal)
        muscular_layout.addWidget(QtWidgets.QLabel('Threshold:'))
        muscular_layout.addWidget(self.m_signal_threshold)
        self.form_layout.addRow('Muscular signal:', muscular_layout)

        # message text edit
        self.message = QtWidgets.QTextEdit()
        self.message.setMaximumHeight(50)
        self.form_layout.addRow('&Message:', self.message)

        # voiceover
        self.voiceover = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Voiceover:', self.voiceover)

        # split record (CSP)
        self.split_checkbox = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Add half time\nextra message (for CSP):', self.split_checkbox)
        self.message2 = QtWidgets.QTextEdit()
        self.message2.setMaximumHeight(50)
        self.form_layout.addRow('&Half time extra message:', self.message2)
        self.split_checkbox.stateChanged.connect(lambda: self.message2.setEnabled(self.split_checkbox.isChecked()))
        self.message2.setEnabled(False)

        # reward settings
        self.reward_signal = QtWidgets.QComboBox()
        self.form_layout.addRow('&Reward signal:', self.reward_signal)
        self.reward_threshold = QtWidgets.QDoubleSpinBox()
        self.reward_threshold.setRange(-10000, 10000)
        self.form_layout.addRow('&Reward threshold:', self.reward_threshold)
        self.show_reward = QtWidgets.QCheckBox()
        self.form_layout.addRow('&Show reward:', self.show_reward)

        # video path
        self.video_path = FileSelectorLine()
        self.form_layout.addRow('&Video file:', self.video_path)

        # ok button
        self.save_button = QtWidgets.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

        self.setMaximumWidth(self.layout().sizeHint().width())

    def handle_enable_mock_previous(self):
        self.mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.random_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.reverse_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.set_enabled_mock_settings()

    def handle_random_mock_previous(self):
        self.mock_previous.setEnabled(not self.random_mock_previous.isChecked())
        #self.reverse_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        #self.set_enabled_mock_settings()

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
        flag = self.type.currentText() == 'Feedback'
        self.mock_file.setEnabled(flag and not self.enable_mock_previous.isChecked())
        self.mock_dataset.setEnabled(flag and not self.enable_mock_previous.isChecked())
        self.mock_previous.setEnabled(flag and self.enable_mock_previous.isChecked())
        self.random_mock_previous.setEnabled(flag and self.enable_mock_previous.isChecked())
        self.enable_mock_previous.setEnabled(flag)
        self.reverse_mock_previous.setEnabled(flag and self.enable_mock_previous.isChecked())

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
        self.random_over_time.setValue(current_protocol['fRandomOverTime'])
        self.update_statistics.setChecked(current_protocol['bUpdateStatistics'])
        self.update_statistics_type.setCurrentIndex(['meanstd', 'max'].index(current_protocol['sStatisticsType']))
        self.update_statistics_type.setEnabled(self.update_statistics.isChecked())
        self.beep_after.setChecked(current_protocol['bBeepAfter'])
        self.auto_bci_fit.setChecked(current_protocol['bAutoBCIFit'])
        self.mock_source.setChecked(current_protocol['bMockSource'])
        self.pause_after.setChecked(current_protocol['bPauseAfter'])
        self.detection_task.setChecked(current_protocol['bEnableDetectionTask'])
        self.drop_outliers.setValue(current_protocol['iDropOutliers'])
        self.drop_outliers.setEnabled(self.update_statistics.isChecked())
        self.ssd_in_the_end.setChecked(current_protocol['bSSDInTheEnd'])
        self.source_signal.setCurrentIndex(
            self.source_signal.findText(current_protocol['fbSource'], QtCore.Qt.MatchFixedString))
        current_protocol_type = ('Feedback' if current_protocol['sFb_type'] == 'CircleFeedback'
                                 else current_protocol['sFb_type'])
        self.type.setCurrentIndex(
            self.type.findText(current_protocol_type, QtCore.Qt.MatchFixedString))
        self.circle_border.setCurrentIndex(current_protocol['iRandomBound'])
        self.circle_border.setEnabled(self.type.currentText() == 'Feedback')
        self.blink_duration_ms.setValue(current_protocol['fBlinkDurationMs'])
        self.blink_threshold.setValue(current_protocol['fBlinkThreshold'])
        self.mock_file.path.setText(current_protocol['sMockSignalFilePath'])
        self.mock_dataset.setText(current_protocol['sMockSignalFileDataset'])
        self.message.setText(current_protocol['cString'])
        self.message2.setText(current_protocol['cString2'])
        self.voiceover.setChecked(current_protocol['bVoiceover'])
        self.split_checkbox.setChecked(current_protocol['bUseExtraMessage'])
        current_index = self.reward_signal.findText(current_protocol['sRewardSignal'], QtCore.Qt.MatchFixedString)
        self.reward_signal.setCurrentIndex(current_index if current_index > -1 else 0)
        self.show_reward.setChecked(current_protocol['bShowReward'])
        self.reward_threshold.setValue(current_protocol['bRewardThreshold'])
        self.enable_mock_previous.setChecked(current_protocol['iMockPrevious'] > 0)
        self.mock_previous.setValue(current_protocol['iMockPrevious'])
        self.random_mock_previous.setChecked(current_protocol['bRandomMockPrevious'])
        self.reverse_mock_previous.setChecked(current_protocol['bReverseMockPrevious'])
        self.mock_previous.setEnabled(self.enable_mock_previous.isChecked()
                                      and not self.random_mock_previous.isChecked())
        self.random_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.reverse_mock_previous.setEnabled(self.enable_mock_previous.isChecked())
        self.video_path.path.setText(current_protocol['sVideoPath'])
        self.video_path.setEnabled(self.type.currentText() == 'Video')
        signals = ([d['sSignalName'] for d in self.parent().parent().params['vSignals']['DerivedSignal']] +
                   [d['sSignalName'] for d in self.parent().parent().params['vSignals']['CompositeSignal']])
        self.m_signal.addItems(['None'] + signals)
        self.m_signal.setEnabled(self.type.currentText() == 'Feedback')
        self.m_signal_threshold.setEnabled(self.type.currentText() == 'Feedback')
        current_index = self.m_signal.findText(current_protocol['sMSignal'], QtCore.Qt.MatchFixedString)
        self.m_signal.setCurrentIndex(current_index if current_index > -1 else 0)
        self.m_signal_threshold.setValue(current_protocol['fMSignalThreshold'])
        pass

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sProtocolName'] = self.name.text()
        self.params[current_signal_index]['fDuration'] = self.duration.value()
        self.params[current_signal_index]['fRandomOverTime'] = self.random_over_time.value()
        self.params[current_signal_index]['bUpdateStatistics'] = int(self.update_statistics.isChecked())
        self.params[current_signal_index]['sStatisticsType'] = self.update_statistics_type.currentText()
        self.params[current_signal_index]['bBeepAfter'] = int(self.beep_after.isChecked())
        self.params[current_signal_index]['bAutoBCIFit'] = int(self.auto_bci_fit.isChecked())
        self.params[current_signal_index]['bMockSource'] = int(self.mock_source.isChecked())
        self.params[current_signal_index]['bPauseAfter'] = int(self.pause_after.isChecked())
        self.params[current_signal_index]['bEnableDetectionTask'] = int(self.detection_task.isChecked())
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
        self.params[current_signal_index]['bVoiceover'] = int(self.voiceover.isChecked())
        self.params[current_signal_index]['bUseExtraMessage'] = int(self.split_checkbox.isChecked())
        self.params[current_signal_index]['sRewardSignal'] = self.reward_signal.currentText()
        self.params[current_signal_index]['bShowReward'] = int(self.show_reward.isChecked())
        self.params[current_signal_index]['bRewardThreshold'] = self.reward_threshold.value()
        self.params[current_signal_index]['iMockPrevious'] = (
            self.mock_previous.value() if self.enable_mock_previous.isChecked() else 0)
        self.params[current_signal_index]['bReverseMockPrevious'] = (
            int(self.reverse_mock_previous.isChecked()) if self.enable_mock_previous.isChecked() else 0)
        self.params[current_signal_index]['bRandomMockPrevious'] = (
            int(self.random_mock_previous.isChecked()) if self.enable_mock_previous.isChecked() else 0)
        self.params[current_signal_index]['sVideoPath'] = self.video_path.path.text()
        self.params[current_signal_index]['sMSignal'] = self.m_signal.currentText()
        self.params[current_signal_index]['fMSignalThreshold'] = self.m_signal_threshold.value()
        self.parent().reset_items()
        self.close()
