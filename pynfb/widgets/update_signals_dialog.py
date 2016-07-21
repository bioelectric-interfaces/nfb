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
        self.columns = ['Signal', 'Modified', 'Band', 'Rejections', 'Spatial filter', 'Open SSD']
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(signals))
        self.setHorizontalHeaderLabels(self.columns)

        # set ch names
        for ind, signal in enumerate(signals):

            # name
            name_item = QtGui.QTableWidgetItem(signal.name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(ind, self.columns.index('Signal'), name_item)

            # status
            modified_item = QtGui.QTableWidgetItem('No')
            modified_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(ind, self.columns.index('Modified'), modified_item)

            # band
            band_widget = BandWidget()
            band_widget.set_band(signal.bandpass)
            self.setCellWidget(ind, self.columns.index('Band'), band_widget)

            # rejection
            self.setItem(ind, self.columns.index('Rejections'), QtGui.QTableWidgetItem('0'))

            # spatial filter
            text = 'Zeros' if signal.spatial_filter_is_zeros() else 'Not trivial'
            self.setItem(ind, self.columns.index('Spatial filter'), QtGui.QTableWidgetItem(text))

        # open buttons
        self.buttons = []
        for ind, _w in enumerate(self.names):
            open_ssd_btn = QtGui.QPushButton('Open')
            open_ssd_btn.clicked.connect(lambda: self.set_modified(ind))
            self.buttons.append(open_ssd_btn)
            self.setCellWidget(ind, self.columns.index('Open SSD'), open_ssd_btn)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.resizeColumnsToContents()

    def set_modified(self, ind):
        row = self.buttons.index(self.sender())
        print('CURREEENNNNT', row)
        self.item(row, self.columns.index('Modified')).setText('Yes')
        self.current_row = row


class BandWidget(QtGui.QWidget):
    def __init__(self, max_freq=10000, **kwargs):
        super(BandWidget, self).__init__(**kwargs)
        layout = QtGui.QHBoxLayout(self)
        layout.setMargin(0)
        self.left = QtGui.QDoubleSpinBox()
        self.left.setMinimumHeight(25)
        self.left.setRange(0, max_freq)
        self.right = QtGui.QDoubleSpinBox()
        self.right.setRange(0, max_freq)
        self.right.setMinimumHeight(25)
        layout.addWidget(self.left)
        layout.addWidget(QtGui.QLabel('-'))
        layout.addWidget(self.right)
        layout.addWidget(QtGui.QLabel('Hz '))

    def set_band(self, band=(0, 0)):
        self.left.setValue(band[0])
        self.right.setValue(band[1])

    def get_band(self):
        return self.left.value(), self.right.value()



class SignalsSSDManager(QtGui.QDialog):
    def __init__(self, signals, x, pos, channels_names, sampling_freq=1000, message=None, **kwargs):
        super(SignalsSSDManager, self).__init__(**kwargs)

        # attributes
        self.signals = [signal for signal in signals if isinstance(signal, DerivedSignal)]
        self.x = x
        self.pos = pos
        self.channels_names = channels_names
        self.sampling_freq = sampling_freq

        #layout
        layout = QtGui.QVBoxLayout(self)
        self.setMinimumWidth(500)

        # table
        self.table = Table(self.signals)
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
    from pynfb.signals import CompositeSignal
    signals = [DerivedSignal(name='Signal'+str(k), bandpass_low=0+k, bandpass_high=1+10*k, spatial_filter=np.array([k])) for k in range(3)]
    signals +=[CompositeSignal(signals, '', 'Composite')]
    app = QtGui.QApplication([])
    w = SignalsSSDManager(signals, 0, 0, 0)
    w.show()
    app.exec_()