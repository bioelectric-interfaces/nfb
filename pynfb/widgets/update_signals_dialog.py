from PyQt4 import QtGui, QtCore
import sys
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas


class Table(QtGui.QTableWidget):
    def __init__(self, signals, *args):
        super(Table, self).__init__(*args)

        self.names = [signal.name for signal in signals]

        # set size and names
        self.setColumnCount(3)
        self.setRowCount(len(signals))
        self.setHorizontalHeaderLabels(['Signal name', 'Open SSD', 'Modified'])

        # set ch names
        for ind, name in enumerate(self.names):
            name_item = QtGui.QTableWidgetItem(name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(ind, 0, name_item)
            modified_item = QtGui.QTableWidgetItem('No')
            modified_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(ind, 2, modified_item)

        # open buttons
        self.weights = [0 for _j in range(self.rowCount())]
        for ind, w in enumerate(self.weights):
            open_ssd_btn = QtGui.QPushButton('Open')
            open_ssd_btn.clicked.connect(lambda : self.set_modified(ind))
            self.setCellWidget(ind, 1, open_ssd_btn)

        # formatting
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def set_modified(self, ind):
        row = [self.cellWidget(i, 1) for i in range(self.rowCount())].index(self.sender())
        self.item(row, 2).setText('Yes')


class SignalsSSDManager(QtGui.QDialog):
    def __init__(self, signals, message=None, **kwargs):
        super(SignalsSSDManager, self).__init__(**kwargs)
        layout = QtGui.QVBoxLayout(self)

        # table
        self.table = Table(signals)
        layout.addWidget(self.table)

        # message
        if message is not None:
            layout.addWidget(QtGui.QLabel(message))

        # ok button
        ok_button = QtGui.QPushButton('OK')
        ok_button.clicked.connect(self.ok_button_action)
        ok_button.setMaximumWidth(100)
        layout.addWidget(ok_button)

    def ok_button_action(self):
        self.close()

    @staticmethod
    def run(signals, message=None):
        selector = SignalsSSDManager(signals, message=message)
        _result = selector.exec_()

if __name__ == '__main__':
    import numpy as np
    from pynfb.signals import DerivedSignal
    signals = [DerivedSignal(name='Signal'+str(k)) for k in range(3)]
    app = QtGui.QApplication([])
    SignalsSSDManager.run(signals, 'Run it')