from copy import deepcopy

from PyQt4 import QtGui, QtCore
import sys

from pynfb.protocols import SelectSSDFilterWidget
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from pynfb.protocols.user_inputs import SelectCSPFilterWidget
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup
from pynfb.signals import DerivedSignal
from numpy import dot

class Table(QtGui.QTableWidget):
    def __init__(self, signals, *args):
        super(Table, self).__init__(*args)
        self.signals = signals
        self.names = [signal.name for signal in signals]

        # set size and names
        self.columns = ['Signal', 'Modified', 'Band', 'Rejections', 'Drop rejections', 'Spatial filter', 'Open SSD',
                        'Open CSP', 'Open ICA']
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(signals))
        self.setHorizontalHeaderLabels(self.columns)

        # set ch names
        for ind, signal in enumerate(signals):

            # name
            name_item = QtGui.QTableWidgetItem(signal.name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(ind, self.columns.index('Signal'), name_item)
            self.update_row(ind)


        # buttons
        self.buttons = []
        self.drop_rejections_buttons = []
        self.csp_buttons = []
        self.ica_buttons = []
        for ind, _w in enumerate(self.names):
            open_ssd_btn = QtGui.QPushButton('Open')
            self.buttons.append(open_ssd_btn)
            self.setCellWidget(ind, self.columns.index('Open SSD'), open_ssd_btn)
            btn = QtGui.QPushButton('Open')
            self.csp_buttons.append(btn)
            self.setCellWidget(ind, self.columns.index('Open CSP'), btn)
            btn = QtGui.QPushButton('Open')
            self.ica_buttons.append(btn)
            self.setCellWidget(ind, self.columns.index('Open ICA'), btn)
            save_btn = QtGui.QPushButton('Drop')
            self.drop_rejections_buttons.append(save_btn)
            self.setCellWidget(ind, self.columns.index('Drop rejections'), save_btn)


        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.resizeColumnsToContents()

    def update_row(self, ind, modified=False):
        signal = self.signals[ind]
        # status
        modified_item = QtGui.QTableWidgetItem('Yes' if modified else 'No')
        modified_item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.setItem(ind, self.columns.index('Modified'), modified_item)

        # band
        band_widget = BandWidget()
        band_widget.set_band(signal.bandpass)
        self.setCellWidget(ind, self.columns.index('Band'), band_widget)

        # rejection
        n_rejections = len(signal.rejections)
        if signal.ica_rejection is not None:
            n_rejections += 1
        self.setItem(ind, self.columns.index('Rejections'), QtGui.QTableWidgetItem(str(n_rejections)))

        # spatial filter
        text = 'Zeros' if signal.spatial_filter_is_zeros() else 'Not trivial'
        self.setItem(ind, self.columns.index('Spatial filter'), QtGui.QTableWidgetItem(text))

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
    test_signal = QtCore.pyqtSignal()
    test_closed_signal = QtCore.pyqtSignal()
    def __init__(self, signals, x, pos, channels_names, protocol, signals_rec, protocols, sampling_freq=1000,
                 message=None, **kwargs):
        super(SignalsSSDManager, self).__init__(**kwargs)

        # attributes
        self.signals = [signal for signal in signals if isinstance(signal, DerivedSignal)]
        self.init_signals = deepcopy(self.signals)
        self.all_signals = signals
        self.x = x
        self.pos = pos
        self.channels_names = channels_names
        self.sampling_freq = sampling_freq
        self.protocol = protocol
        self.signals_rec = signals_rec
        self.stats = [(signal.mean, signal.std, signal.scaling_flag) for signal in signals]
        self.ica = None

        #layout
        layout = QtGui.QVBoxLayout(self)
        self.setMinimumWidth(750)

        # table
        self.table = Table(self.signals)
        layout.addWidget(self.table)

        # message
        if message is not None:
            layout.addWidget(QtGui.QLabel(message))

        # bottom layout
        bottom_layout = QtGui.QHBoxLayout()
        layout.addLayout(bottom_layout)

        # ok button
        self.ok_button = QtGui.QPushButton('Continue')
        self.ok_button.clicked.connect(self.ok_button_action)
        self.ok_button.setMaximumWidth(100)
        self.ok_button.setMinimumHeight(25)

        # revert changes
        self.revert_button = QtGui.QPushButton('Revert changes')
        self.revert_button.clicked.connect(self.revert_changes)
        self.revert_button.setMaximumWidth(100)
        self.revert_button.setMinimumHeight(25)

        # test protocol
        self.test_button = QtGui.QPushButton('Test')
        self.test_button.clicked.connect(self.test_action)
        self.test_button.setMaximumWidth(100)
        self.test_button.setMinimumHeight(25)

        self.combo_protocols = QtGui.QComboBox()
        protocols_names = [prot.name for prot in protocols]
        self.combo_protocols.addItems(protocols_names)

        # add to bottom layout
        bottom_layout.addWidget(self.test_button)
        bottom_layout.addWidget(self.combo_protocols)
        bottom_layout.addWidget(self.revert_button)
        bottom_layout.addWidget(self.ok_button)

        #self.test_button.hide()


        for j, button in enumerate(self.table.buttons):
            button.clicked.connect(lambda: self.run_ssd())
            button.setEnabled(isinstance(self.signals[j], DerivedSignal))

        for j, button in enumerate(self.table.csp_buttons):
            button.clicked.connect(lambda: self.run_ssd(csp=True))
            button.setEnabled(isinstance(self.signals[j], DerivedSignal))

        for j, button in enumerate(self.table.ica_buttons):
            button.clicked.connect(lambda: self.run_ssd(ica=True))
            button.setEnabled(isinstance(self.signals[j], DerivedSignal))

        for j, button in enumerate(self.table.drop_rejections_buttons):
            button.clicked.connect(lambda: self.drop_rejections())
            button.setEnabled(isinstance(self.signals[j], DerivedSignal))


    def test_action(self):
        if self.test_button.text() == 'Test':
            print('Test run')
            self.test_button.setText('Close test')
            self.revert_button.setDisabled(True)
            self.ok_button.setDisabled(True)
            self.combo_protocols.setDisabled(True)
            self.protocol.update_mean_std(self.x, self.signals_rec, must=True)
            #self.main.update_statistics_lines()
            self.test_signal.emit()
            self.setModal(False)
        else:
            print('Test close')
            self.test_button.setText('Test')
            self.revert_button.setEnabled(True)
            self.ok_button.setEnabled(True)
            self.combo_protocols.setEnabled(True)
            for j, (mean, std, flag) in enumerate(self.stats):
                self.all_signals[j].mean = mean
                self.all_signals[j].std = std
                self.all_signals[j].scaling_flag = flag
            self.test_closed_signal.emit()
            self.setModal(False)
        # self.close()

    def revert_changes(self):
        quit_msg = "Are you sure you want to revert all changes?"
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            for j, signal in enumerate(self.signals):
                signal.update_rejections(self.init_signals[j].rejections, append=False)
                signal.update_ica_rejection(rejection=self.init_signals[j].ica_rejection)
                signal.update_spatial_filter(self.init_signals[j].spatial_filter)
                signal.update_bandpass(self.init_signals[j].bandpass)
                self.table.update_row(j, modified=False)

    def drop_rejections(self):
        row = self.table.drop_rejections_buttons.index(self.sender())
        if len(self.signals[row].rejections) > 0 or self.signals[row].rejections is not None:
            quit_msg = "Are you sure you want to drop {} rejections of signal \"{}\"?".format(
                len(self.signals[row].rejections),
                self.signals[row].name)
            reply = QtGui.QMessageBox.question(self, 'Message',
                                               quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                self.signals[row].update_rejections(rejections=[], append=False)
                self.signals[row].update_ica_rejection(rejection=None)
                self.table.update_row(row, modified=True)

    def run_ssd(self, row=None, csp=False, ica=False):
        if row is None and ica:
            row = self.table.ica_buttons.index(self.sender())
        elif row is None and not csp:
            row = self.table.buttons.index(self.sender())
        elif row is None and csp:
            row = self.table.csp_buttons.index(self.sender())

        x = self.x
        if self.signals[row].ica_rejection is not None:
            x = dot(x, self.signals[row].ica_rejection)
        for rejection in self.signals[row].rejections:
            x = dot(x, rejection)

        if ica:
            reply = QtGui.QMessageBox.Yes
            if len(self.signals[row].rejections) > 0:
                reply = QtGui.QMessageBox.question(self, 'Message',
                                                   'Rejections already exist. Are you sure you want to do ICA?'
                                                   ' (ICA should be the first)',
                                                   QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                ica_rejection, self.ica = ICADialog.get_rejection(x, self.channels_names, self.sampling_freq,
                                                                  ica=self.ica)
                self.signals[row].update_ica_rejection(ica_rejection)
            rejections = []
            filter = None
            bandpass = None
        else:
            SelectFilterWidget = SelectCSPFilterWidget if csp else SelectSSDFilterWidget
            filter, bandpass, rejections = SelectFilterWidget.select_filter_and_bandpass(x, self.pos,
                                                                                            self.channels_names,
                                                                                            sampling_freq=
                                                                                        self.sampling_freq)
        if filter is not None:
            self.signals[row].update_spatial_filter(filter)

        if bandpass is not None:
            self.signals[row].update_bandpass(bandpass)

        self.signals[row].update_rejections(rejections, append=True)

        modified_flag = len(rejections)>0 or bandpass is not None or filter is not None
        self.table.update_row(row, modified=modified_flag)


    def ok_button_action(self):
        for row in range(self.table.rowCount()):
            band = self.table.cellWidget(row, self.table.columns.index('Band')).get_band()
            self.signals[row].update_bandpass(band)
        self.close()




if __name__ == '__main__':
    import numpy as np
    from pynfb.signals import CompositeSignal
    signals = [DerivedSignal(ind = k, name='Signal'+str(k), bandpass_low=0+k, bandpass_high=1+10*k, spatial_filter=np.array([k]), n_channels=4) for k in range(3)]
    signals +=[CompositeSignal(signals, '', 'Composite', 3)]
    app = QtGui.QApplication([])

    x = np.random.randn(1000, 4)
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    channels = ['Cz', 'Fp1', 'Fp2', 'Pz']

    w = SignalsSSDManager(signals, x, ch_names_to_2d_pos(channels), channels, None, None, [])
    w.show()
    app.exec_()