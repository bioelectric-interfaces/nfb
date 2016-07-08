from PyQt4 import QtGui

from pynfb.io.defaults import vectors_defaults as defaults
from pynfb.settings_widget import FileSelectorLine

default_signal = defaults['vSignals']['CompositeSignal'][0]


class CompositeSignalsSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vSignals']['CompositeSignal']

        # layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        # label
        label = QtGui.QLabel('Composite signals:')
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
        self.params = self.parent().params['vSignals']['CompositeSignal']
        self.list.clear()
        self.signals_dialogs = []
        print(self.params)
        for signal in self.params:
            print(signal)
            item = QtGui.QListWidgetItem(signal['sSignalName'])
            self.signals_dialogs.append(CompositeSignalDialog(self, signal_name=signal['sSignalName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setItemSelected(self.list.item(0), True)


class CompositeSignalDialog(QtGui.QDialog):
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

        # operation type combo box:
        self.expression = QtGui.QLineEdit()
        self.expression.setMaximumHeight(50)
        self.form_layout.addRow('&Expression:', self.expression)

        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def open(self):
        self.reset_items()
        super().open()

    def reset_items(self):
        current_signal_index = self.parent().list.currentRow()
        self.expression.setText(self.params[current_signal_index]['sExpression'])

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sSignalName'] = self.name.text()
        self.params[current_signal_index]['sExpression'] = self.expression.text()
        self.parent().reset_items()
        self.close()