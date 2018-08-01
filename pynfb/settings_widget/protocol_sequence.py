from PyQt5 import QtCore, QtGui, QtWidgets


class ProtocolSequenceSettingsWidget(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params['vPSequence']
        label = QtWidgets.QLabel('Protocols sequence:')
        self.list = ProtocolSequenceListWidget(parent=self)
        # self.list.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        # self.list.connect.
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        buttons_layout = QtWidgets.QHBoxLayout()
        remove_signal_button = QtWidgets.QPushButton('Remove')
        remove_signal_button.clicked.connect(self.list.remove_current_row)
        buttons_layout.addWidget(remove_signal_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def reset_items(self):
        self.params = self.parent().params['vPSequence']
        self.list.reset_items()


class ProtocolSequenceListWidget(QtWidgets.QListWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.reset_items()

    def dropEvent(self, QDropEvent):
        super().dropEvent(QDropEvent)
        self.save()

    def reset_items(self):
        self.params = self.parent().params

        self.clear()
        for protocol in self.params:
            item = QtWidgets.QListWidgetItem(protocol)
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
