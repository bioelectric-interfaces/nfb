from PyQt4 import QtGui

inlet_types = ['lsl', 'lsl_from_file', 'lsl_generator', 'ftbuffer']


class InletSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.combo = QtGui.QComboBox()
        self.combo.addItem('LSL stream')
        self.combo.addItem('LSL from file')
        self.combo.addItem('LSL generator')
        self.combo.addItem('Field Trip buffer')
        self.line_edit_1 = QtGui.QLineEdit()
        self.line_edit_1.textChanged.connect(self.line_edit_1_changed_event)
        self.line_edit_2 = QtGui.QLineEdit()
        self.line_edit_2.textChanged.connect(self.line_edit_2_changed_event)
        # self.stream_name = QtGui.QLineEdit()
        # self.stream_name.textChanged.connect(self.stream_name_changed_event)
        # self.raw_path = QtGui.QLineEdit('')
        # self.raw_path.textChanged.connect(self.raw_path_changed_event)
        self.raw_select_button = QtGui.QPushButton('Select file...')
        self.raw_select_button.clicked.connect(self.chose_file_action)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.combo)
        layout.addWidget(self.line_edit_1)
        layout.addWidget(self.line_edit_2)
        # layout.addWidget(self.stream_name)
        # layout.addWidget(self.raw_path)
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
            self.parent().params['sFTHostnamePort'] = self.line_edit_1.text() + ':' + port

    def combo_changed_event(self):
        self.parent().params['sInletType'] = inlet_types[self.combo.currentIndex()]
        self.raw_select_button.hide()
        self.line_edit_1.setEnabled(True)
        self.line_edit_2.hide()
        if self.combo.currentIndex() == 0:
            self.line_edit_1.setPlaceholderText('Print LSL stream name')
            self.line_edit_1.setText(self.parent().params['sStreamName'])
        elif self.combo.currentIndex() == 1:
            self.raw_select_button.show()
            self.line_edit_1.setPlaceholderText('Print raw data file path')
            self.line_edit_1.setText(self.parent().params['sRawDataFilePath'])
        elif self.combo.currentIndex() == 2:
            self.line_edit_1.setPlaceholderText('')
            self.line_edit_1.setEnabled(False)
            self.line_edit_1.setText('')
        elif self.combo.currentIndex() == 3:
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

    def reset(self):
        self.combo.setCurrentIndex(inlet_types.index(self.parent().params['sInletType']))
        self.combo_changed_event()