from PyQt4 import QtCore

from PyQt4 import QtGui

from pynfb.io.defaults import vectors_defaults as defaults
from pynfb.settings_widget import FileSelectorLine

default_signal = defaults['vPGroups']['PGroup'][0]


class ProtocolGroupsSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vPGroups']['PGroup']


        # layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        # label
        label = QtGui.QLabel('Protocol groups:')
        layout.addWidget(label)

        # list of signals
        self.list = QtGui.QListWidget(self)
        self.list.setDragDropMode(QtGui.QAbstractItemView.DragOnly)
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
        self.params = self.parent().params['vPGroups']['PGroup']
        self.list.clear()
        self.signals_dialogs = []
        print(self.params)
        for signal in self.params:
            print(signal)
            item = QtGui.QListWidgetItem(signal['sName'])
            self.signals_dialogs.append(ProtocolGroupDialog(self, signal_name=signal['sName']))
            self.list.addItem(item)
        if self.list.currentRow() < 0:
            self.list.setItemSelected(self.list.item(0), True)


class ProtocolGroupDialog(QtGui.QDialog):
    def __init__(self, parent, signal_name='Signal'):
        self.params = parent.params
        super().__init__(parent)
        self.parent_list = parent
        self.setWindowTitle('Properties: ' + signal_name)
        self.form_layout = QtGui.QFormLayout(self)

        # name
        self.name = QtGui.QLineEdit(self)
        self.name.setText(signal_name)
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[a-zA-Z0-9_]+$"))
        self.name.setValidator(validator)
        self.form_layout.addRow('&Name:', self.name)

        # operation type combo box:
        self.expression = QtGui.QLineEdit()
        self.expression.setMaximumHeight(50)
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[a-zA-Z0-9_ ]+$"))
        self.expression.setValidator(validator)
        self.form_layout.addRow('&Protocols\n(separated by space):', self.expression)

        # operation type combo box:
        self.numbers = QtGui.QLineEdit()
        self.numbers.setMaximumHeight(50)
        validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9 ]+$"))
        self.numbers.setValidator(validator)
        self.form_layout.addRow('&Corresponding quantity\n(separated by space):', self.numbers)

        # shuffle
        self.shuffle = QtGui.QCheckBox('Shuffle')
        self.form_layout.addRow('&Order:', self.shuffle)

        # ok button
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save_and_close)
        self.form_layout.addRow(self.save_button)

    def open(self):
        self.reset_items()
        super().open()

    def reset_items(self):
        current_signal_index = self.parent().list.currentRow()
        self.expression.setText(str(self.params[current_signal_index]['sList']))
        self.numbers.setText(str(self.params[current_signal_index]['sNumberList']))
        self.shuffle.setChecked(bool(self.params[current_signal_index]['bShuffle']))

    def save_and_close(self):
        current_signal_index = self.parent().list.currentRow()
        self.params[current_signal_index]['sName'] = str(self.name.text())
        self.params[current_signal_index]['sList'] = str(self.expression.text())
        self.params[current_signal_index]['sNumberList'] = str(self.numbers.text())
        self.params[current_signal_index]['bShuffle'] = int(self.shuffle.isChecked())
        self.parent().reset_items()
        self.close()

