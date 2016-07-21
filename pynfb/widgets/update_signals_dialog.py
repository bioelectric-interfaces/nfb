from PyQt4 import QtGui, QtCore
import sys

from pynfb.protocols import SelectSSDFilterWidget
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup
from pynfb.signals import DerivedSignal


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
        self.buttons = []
        for ind, _w in enumerate(self.names):
            open_ssd_btn = QtGui.QPushButton('Open')
            open_ssd_btn.clicked.connect(lambda: self.set_modified(ind))
            self.buttons.append(open_ssd_btn)
            self.setCellWidget(ind, 1, open_ssd_btn)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def set_modified(self, ind):
        row = self.buttons.index(self.sender())
        print('CURREEENNNNT', row)
        self.item(row, 2).setText('Yes')
        self.current_row = row


class SignalsSSDManager(QtGui.QDialog):
    def __init__(self, signals, x, pos, channels_names, sampling_freq=1000, message=None, **kwargs):
        super(SignalsSSDManager, self).__init__(**kwargs)

        # attributes
        self.signals = signals
        self.x = x
        self.pos = pos
        self.channels_names = channels_names
        self.sampling_freq = sampling_freq

        #layout
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

        for j, button in enumerate(self.table.buttons):
            button.clicked.connect(lambda: self.run_ssd())

            button.setEnabled(isinstance(self.signals[j], DerivedSignal))
            

    def run_ssd(self, row=None):
        if row is None:
            row = self.table.buttons.index(self.sender())

        filter, bandpass = SelectSSDFilterWidget.select_filter_and_bandpass(self.x, self.pos, self.channels_names,
                                                                            sampling_freq=self.sampling_freq)
        if filter is not None:
            if filter.ndim == 1:
                self.signals[row].update_spatial_filter(filter)
            else:
                self.signals[row].append_spatial_matrix_list(filter)
                filter = SpatialFilterSetup.get_filter(self.channels_names,
                                                       message='Current spatial filter for signal is null vector. '
                                                               'Please modify it.')
                self.signals[row].update_spatial_filter(filter)
        if bandpass is not None:
            self.signals[row].update_bandpass(bandpass)
        
        # emulate signal with new spatial filter
        signal = self.signals[row]
        signal.reset_statistic_acc()
        mean_chunk_size = 8
        for k in range(0, self.x.shape[0] - mean_chunk_size, mean_chunk_size):
            chunk = self.x[k:k + mean_chunk_size]
            signal.update(chunk)

    def ok_button_action(self):
        self.close()




if __name__ == '__main__':
    import numpy as np
    signals = [DerivedSignal(name='Signal'+str(k)) for k in range(3)]
    app = QtGui.QApplication([])
    w = SignalsSSDManager(signals, 0, 0, 0)
    w.show()
    app.exec_()