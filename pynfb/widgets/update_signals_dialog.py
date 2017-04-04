from copy import deepcopy

from PyQt4 import QtGui, QtCore
import sys

from pynfb.protocols import SelectSSDFilterWidget
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from pynfb.widgets.rejections_editor import RejectionsWidget
from pynfb.widgets.spatial_filter_setup import SpatialFilterSetup
from pynfb.widgets.check_table import CheckTable
from pynfb.signals import DerivedSignal
from numpy import dot, concatenate, array

class SignalsTable(QtGui.QTableWidget):
    show_topography_name = {True: 'Topography', False: 'Filter'}

    def __init__(self, signals, channels_names, *args):
        super(SignalsTable, self).__init__(*args)
        self.signals = signals
        self.names = [signal.name for signal in signals]
        self.channels_names = channels_names


        # set size and names
        self.columns = ['Signal', 'Band', 'Rejections', 'Spatial filter', 'SSD', 'CSP', 'ICA']
        self.columns_width = [80, 150, 150, 80, 50, 50, 50]
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(signals))
        self.setHorizontalHeaderLabels(self.columns)

        # set ch names
        self.show_topography = []
        for ind, signal in enumerate(signals):
            # show topography flag
            self.show_topography.append(False)

            # name
            name_item = QtGui.QTableWidgetItem(signal.name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            name_item.setTextAlignment(QtCore.Qt.AlignCenter)
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
            self.setCellWidget(ind, self.columns.index('SSD'), open_ssd_btn)
            btn = QtGui.QPushButton('Open')
            self.csp_buttons.append(btn)
            self.setCellWidget(ind, self.columns.index('CSP'), btn)
            btn = QtGui.QPushButton('Open')
            self.ica_buttons.append(btn)
            self.setCellWidget(ind, self.columns.index('ICA'), btn)

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.resizeColumnsToContents()

        for ind, width in enumerate(self.columns_width):
            self.setColumnWidth(ind, width)



    def update_row(self, ind, modified=False):
        signal = self.signals[ind]
        show_topography = self.show_topography[ind]

        # band
        band_widget = BandWidget()
        band_widget.set_band(signal.bandpass)
        self.setCellWidget(ind, self.columns.index('Band'), band_widget)

        # rejection
        if len(signal.rejections) == 0:
            rejections = QtGui.QLabel('Empty')
            rejections.setAlignment(QtCore.Qt.AlignCenter)
        else:
            rejections = RejectionsWidget(self.channels_names, signal_name=self.signals[ind].name)
            rejections.set_rejections(signal.rejections)
            rejections.rejection_deleted.connect(lambda ind: signal.drop_rejection(ind))
        self.setCellWidget(ind, self.columns.index('Rejections'), rejections)

        # spatial filter
        scale = 80
        topo_canvas = TopographicMapCanvas()
        topo_canvas.setMaximumHeight(scale - 2)
        topo_canvas.setMaximumWidth(scale)
        data = self.signals[ind].spatial_filter if not show_topography else self.signals[ind].spatial_filter_topography
        if data is None:
            topo_canvas.draw_central_text("not\nfound", right_bottom_text=self.show_topography_name[show_topography])
        else:
            topo_canvas.update_figure(data, names=self.channels_names, show_names=[],
                                      show_colorbar=False, right_bottom_text=self.show_topography_name[show_topography])
        #text = 'Zeros' if signal.spatial_filter_is_zeros() else 'Not trivial'
        #self.setItem(ind, self.columns.index('Spatial filter'), QtGui.QTableWidgetItem(text))

        self.setCellWidget(ind, self.columns.index('Spatial filter'), topo_canvas)
        self.setRowHeight(ind, scale)

    def contextMenuEvent(self, pos):
        if self.columnAt(pos.x()) == self.columns.index('Spatial filter'):
            self.open_selection_menu(self.rowAt(pos.y()))

    def switch_filter_topography(self, row):
        self.show_topography[row] = not self.show_topography[row]
        for row in range(self.rowCount()):
            self.update_row(row)


    def open_selection_menu(self, row):
        menu = QtGui.QMenu()
        action = QtGui.QAction('Edit', self)
        action.triggered.connect(lambda: self.edit_spatial_filter(row))
        menu.addAction(action)
        action = QtGui.QAction('Set zeros', self)
        action.triggered.connect(lambda: self.edit_spatial_filter(row, set_zeros=True))
        menu.addAction(action)
        action = QtGui.QAction('Show ' + ('topography' if not self.show_topography[row] else 'filter'), self)
        action.triggered.connect(lambda: self.switch_filter_topography(row))
        menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    def edit_spatial_filter(self, row, set_zeros=False):
        signal = self.signals[row]
        if set_zeros:
            filter_ = np.zeros_like(signal.spatial_filter)
        else:
            filter_ = SpatialFilterSetup.get_filter(
                self.channels_names,
                weights=signal.spatial_filter,
                message='Please modify spatial filter for "{}"'.format(signal.name),
                title='"{}" spatial filter'.format(signal.name))
        signal.update_spatial_filter(filter_)
        self.update_row(row, modified=True)

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
                 message=None, protocol_seq=None, **kwargs):
        super(SignalsSSDManager, self).__init__(**kwargs)

        # name
        self.setWindowTitle('Signals manager')

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
        self.ica_unmixing_matrix = None

        #layout
        main_layout = QtGui.QVBoxLayout(self)
        layout = QtGui.QHBoxLayout()
        main_layout.addLayout(layout)

        # table
        self.table = SignalsTable(self.signals, self.channels_names)
        self.setMinimumWidth(sum(self.table.columns_width) + 250)
        self.setMinimumHeight(400)
        layout.addWidget(self.table)

        # protocols seq check table
        protocol_seq_table = CheckTable(protocol_seq, ['State 1\n ', 'State 2\n(CSP)'], 'Protocol')
        protocol_seq_table.setMaximumWidth(200)
        self.get_checked_protocols = lambda: protocol_seq_table.get_checked_rows()

        # message
        if message is not None:
            layout.addWidget(QtGui.QLabel(message))

        # bottom layout
        bottom_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(bottom_layout)

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
        layout.addWidget(protocol_seq_table)
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
                signal.rejections = self.init_signals[j].rejections
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

        ind = self.get_checked_protocols()
        print(ind)
        x = concatenate([self.x[j] for j in ind[0]])
        if csp:
            x = concatenate([x] + [self.x[j] for j in ind[1]])
        x = dot(x, self.signals[row].rejections.get_prod())

        to_all = False
        ica_rejection = None
        topography = None
        if ica:
            reply = QtGui.QMessageBox.Yes
            if len(self.signals[row].rejections) > 0:
                reply = QtGui.QMessageBox.question(self, 'Warning',
                                                   'Changing ICA base selection will '
                                                   'invalidate the current rejections (CSP, SSD). '
                                                   'Are you sure you want to continue?',
                                                   QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                result = ICADialog.get_rejection(x, self.channels_names, self.sampling_freq,
                                                 unmixing_matrix=self.ica_unmixing_matrix)
                ica_rejection, filter, topography, self.ica_unmixing_matrix, bandpass, to_all = result
            rejections = []
        elif csp:
            rejection, filter, topography, _, bandpass, to_all = ICADialog.get_rejection(x, self.channels_names, self.sampling_freq,
                                                                     mode='csp')
            rejections = [rejection] if rejection is not None else []
        else:
            filter, topography, bandpass, rejections = SelectSSDFilterWidget.select_filter_and_bandpass(x, self.pos,
                                                                                         self.channels_names,
                                                                                         sampling_freq=
                                                                                         self.sampling_freq)


        rows = range(len(self.signals)) if to_all else [row]
        print(to_all, rows)
        for row_ in rows:
            if ica_rejection is not None:
                self.signals[row_].update_ica_rejection(ica_rejection)
            if filter is not None:
                self.signals[row_].update_spatial_filter(filter, topography=topography)
            if bandpass is not None:
                self.signals[row_].update_bandpass(bandpass)
            self.signals[row_].update_rejections(rejections, append=True)
            modified_flag = len(rejections)>0 or bandpass is not None or filter is not None
            self.table.update_row(row_, modified=modified_flag)


    def ok_button_action(self):
        for row in range(self.table.rowCount()):
            band = self.table.cellWidget(row, self.table.columns.index('Band')).get_band()
            self.signals[row].update_bandpass(band)
        self.close()




if __name__ == '__main__':
    import numpy as np
    from pynfb.signals import CompositeSignal
    signals = [DerivedSignal(ind = k, source_freq=500, name='Signal'+str(k), bandpass_low=0+k, bandpass_high=1+10*k, spatial_filter=np.array([k]), n_channels=4) for k in range(3)]
    signals +=[CompositeSignal(signals, '', 'Composite', 3)]
    app = QtGui.QApplication([])

    x = np.random.randn(1000, 4)
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    channels = ['Cz', 'Fp1', 'Fp2', 'Pz']

    w = SignalsSSDManager(signals, [x, -x*0.1], ch_names_to_2d_pos(channels), channels, None, None, [], protocol_seq=['One', 'Two'])
    w.show()
    app.exec_()