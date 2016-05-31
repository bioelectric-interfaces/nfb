from PyQt4 import QtGui, QtCore
import sys
import pyqtgraph as pg
from pynfb.protocols import *
from pynfb.protocols_widgets import *
import time
import os
pg.setConfigOptions(antialias=True)
static_path = full_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__))+'/static')

class LSLPlotDataItem(pg.PlotDataItem):
    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y/(max(y) - min(y) + 1e-20)*0.75
        return x, y


class PlayerButtonsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init buttons
        self.start = QtGui.QPushButton('')
        self.restart = QtGui.QPushButton('')
        # set icons
        self.start.setIcon(QtGui.QIcon(static_path+'/imag/play-button.png'))
        self.restart.setIcon(QtGui.QIcon(static_path+'/imag/replay.png'))
        # set size
        self.start.setMinimumHeight(30)
        self.restart.setMinimumHeight(30)
        # add events
        self.start.clicked.connect(self.start_clicked_event)
        self.restart.clicked.connect(self.restart_clicked_event)
        # properties
        self.start.setCheckable(True)
        self.restart.setEnabled(False)
        # init layer
        layer = QtGui.QHBoxLayout()
        self.setLayout(layer)
        layer.addWidget(self.start)
        layer.addWidget(self.restart)
        self.setMaximumWidth(200)
        self.setMinimumWidth(100)

    def start_clicked_event(self):
        self.restart.setEnabled(True)
        if self.start.isChecked():
            self.start.setIcon(QtGui.QIcon(static_path+'/imag/pause.png'))
        else:
            self.start.setIcon(QtGui.QIcon(static_path+'/imag/play-button.png'))

    def restart_clicked_event(self):
        self.start.setChecked(False)
        self.start_clicked_event()
        self.restart.setEnabled(False)



class MainWindow(QtGui.QMainWindow):
    def __init__(self, current_protocol, signals, n_signals=1, parent=None, n_channels=32, experiment_n_samples=None,
                 experiment=None, freq=500):
        super(MainWindow, self).__init__(parent)
        #link to experiment

        self.source_freq = freq
        self.experiment = experiment

        # player panel
        self.player_panel = PlayerButtonsWidget(parent=self)
        self.player_panel.restart.clicked.connect(self.restart_experiment)


        # timer label
        self.timer_label = QtGui.QLabel('tf')

        # derived signals viewers
        signals_layout = pg.GraphicsLayoutWidget(self)
        self.signal_curves = []
        for signal in signals:
            roi_figure = signals_layout.addPlot(labels={'left': signal.name})
            signals_layout.nextRow()
            curve = roi_figure.plot().curve
            self.signal_curves.append(curve)
        self.signals_buffer = np.zeros((8000, n_signals))
        self.signals_curves_x_net = np.linspace(0, 8000 / self.source_freq, 8000 / 8)

        # data recorders
        self.experiment_n_samples = experiment_n_samples
        self.samples_counter = 0
        self.raw_recorder = np.zeros((experiment_n_samples*110//100, n_channels)) * np.nan
        self.signals_recorder = np.zeros((experiment_n_samples*110//100, n_signals)) * np.nan

        # raw data viewer
        self.raw = pg.PlotWidget(self)
        self.n_channels = n_channels
        self.n_samples = 2000
        self.raw_buffer = np.zeros((self.n_samples, self.n_channels))
        self.scaler = 1
        self.curves = []
        self.x_mesh = np.linspace(0, self.n_samples / self.source_freq, self.n_samples)
        self.raw.setYRange(0, min(8, self.n_channels))
        self.raw.setXRange(0, self.n_samples / self.source_freq)
        self.raw.showGrid(x=None, y=True, alpha=1)
        self.plot_raw_chekbox = QtGui.QCheckBox('plot raw')
        self.plot_raw_chekbox.setChecked(True)
        self.autoscale_raw_chekbox = QtGui.QCheckBox('autoscale')
        self.autoscale_raw_chekbox.setChecked(True)
        for i in range(self.n_channels):
            c = LSLPlotDataItem(pen=(i, self.n_channels * 1.3))
            self.raw.addItem(c)
            c.setPos(0, i + 1)
            self.curves.append(c)

        # main window layout
        layout = pg.LayoutWidget(self)
        layout.addWidget(signals_layout, 0, 0, 1, 2)
        layout.addWidget(self.plot_raw_chekbox, 1, 0, 1, 1)
        layout.addWidget(self.autoscale_raw_chekbox, 1, 1, 1, 1)
        layout.addWidget(self.raw, 2, 0, 1, 2)
        layout.addWidget(self.player_panel, 3, 0, 1, 1)
        layout.addWidget(self.timer_label, 3, 1, 1, 1)
        layout.layout.setRowStretch(0, 2)
        layout.layout.setRowStretch(2, 2)
        self.setCentralWidget(layout)

        # main window settings
        self.resize(800, 400)
        self.show()

        # subject window
        self.subject_window = SubjectWindow(self, current_protocol)
        self.subject_window.show()
        self._subject_window_want_to_close = False

        # time counter
        self.time_counter = 0
        self.t0 = time.time()
        self.t = self.t0

    def redraw_signals(self, samples, chunk, samples_counter):
        if self.player_panel.start.isChecked():
            # record
            if self.samples_counter < self.experiment_n_samples:
                self.raw_recorder[self.samples_counter:self.samples_counter+chunk.shape[0]] = chunk[:, :self.n_channels]
                for s, sample in enumerate(samples):
                    self.signals_recorder[self.samples_counter:self.samples_counter + chunk.shape[0], s] = sample
                self.samples_counter += chunk.shape[0]

        # derived signals
        data_buffer = self.signals_buffer
        data_buffer[:-chunk.shape[0]] = data_buffer[chunk.shape[0]:]
        for s, sample in enumerate(samples):
            data_buffer[-chunk.shape[0]:, s] = sample
            self.signal_curves[s].setData(x=self.signals_curves_x_net, y=data_buffer[::8, s])

        # raw signals
        if self.plot_raw_chekbox.isChecked():
            self.raw_buffer[:-chunk.shape[0]] = self.raw_buffer[chunk.shape[0]:]
            self.raw_buffer[-chunk.shape[0]:] = chunk[:, :self.n_channels]
            for i in range(0, self.n_channels, 1):
                self.curves[i].setData(self.x_mesh, self.raw_buffer[:, i] / self.scaler)
            if self.autoscale_raw_chekbox.isChecked() and self.time_counter % 10 == 0:
                self.scaler = 0.8 * self.scaler + 0.2 * (np.max(self.raw_buffer) - np.min(self.raw_buffer)) / 0.75

        # timer
        if self.time_counter % 10 == 0:
            t_curr = time.time()
            self.timer_label.setText('samples:\t{}\ttime:\t{:.1f}\tfps:\t{:.2f}\tchunk size:\t{}'.format(samples_counter, t_curr - self.t0, 1. / (t_curr - self.t) * 10, chunk.shape[0]))
            self.t = t_curr
        self.time_counter += 1

    def restart_experiment(self):
        self.experiment.restart()

    def closeEvent(self, event):
        self._subject_window_want_to_close = True
        self.subject_window.close()
        self.experiment.destroy()
        event.accept()


class SubjectWindow(QtGui.QMainWindow):
    def __init__(self, parent, current_protocol, **kwargs):
        super(SubjectWindow, self).__init__(parent, **kwargs)
        self.resize(500, 500)
        self.current_protocol = current_protocol
        self.figure = ProtocolWidget()
        self.setCentralWidget(self.figure)
        self.current_protocol.widget_painter.prepare_widget(self.figure)

    def update_protocol_state(self, samples, chunk_size=1):
        self.current_protocol.update_state(samples, chunk_size=chunk_size)
        pass

    def change_protocol(self, new_protocol):
        self.current_protocol = new_protocol
        self.figure.clear()
        self.current_protocol.widget_painter.prepare_widget(self.figure)

    def closeEvent(self, event):
        if self.parent().experiment.is_finished or self.parent()._subject_window_want_to_close:
            event.accept()
        else:
            event.ignore()


def main():
    print(static_path)
    app = QtGui.QApplication(sys.argv)
    widget = PlayerButtonsWidget()
    widget.show()
    sys.exit(app.exec_())
    pass


if __name__ == '__main__':
    main()