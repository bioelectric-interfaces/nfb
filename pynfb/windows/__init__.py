import os
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal

from pynfb.brain import SourceSpaceRecontructor
from pynfb.brain import SourceSpaceWidget
from pynfb.helpers.dc_blocker import DCBlocker
from pynfb.protocols.widgets import ProtocolWidget
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.widgets.signals_painter import RawViewer
from pynfb.widgets.topography import TopomapWidget
from ..widgets.signal_viewers import RawSignalViewer, DerivedSignalViewer

pg.setConfigOptions(antialias=True)

static_path = full_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../static')


class LSLPlotDataItem(pg.PlotDataItem):
    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y / (max(y) - min(y) + 1e-20) * 0.75
        return x, y


class PlayerButtonsWidget(QtWidgets.QWidget):
    start_clicked = pyqtSignal()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init buttons
        self.start = QtWidgets.QPushButton('')
        self.restart = QtWidgets.QPushButton('')
        # set icons
        self.start.setIcon(QtGui.QIcon(static_path + '/imag/play-button.png'))
        self.restart.setIcon(QtGui.QIcon(static_path + '/imag/replay.png'))
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
        layer = QtWidgets.QHBoxLayout()
        self.setLayout(layer)
        layer.addWidget(self.start)
        layer.addWidget(self.restart)
        self.setMaximumWidth(200)
        self.setMinimumWidth(100)

        styles = ["background-color: #{}".format(color) for color in ["FFFFFF", "F8FFFB", "F2FFF8", "ECFFF4", "E6FFF1",
                                                                      "DFFFED", "D9FFEA", "D3FFE6", "CDFFE3", "C7FFE0"]]
        styles += styles[::-1]

        # animation doesn't work for strings but provides an appropriate delay
        animation = QtCore.QPropertyAnimation(self.start, b'styleSheet')
        animation.setDuration(40)

        states = [QtCore.QState() for style in styles]
        for j, style in enumerate(styles):
            states[j].assignProperty(self.start, 'styleSheet', style)
            states[j].addTransition(states[j].propertiesAssigned, states[(j+1) % len(styles)])
        self.init_state = states[0]


        self.machine = QtCore.QStateMachine()
        self.machine.addDefaultAnimation(animation)
        for state in states:
            self.machine.addState(state)
        self.machine.setInitialState(states[0])
        self.machine.start()

    def start_clicked_event(self):
        self.restart.setEnabled(True)
        if self.start.isChecked():
            self.start_clicked.emit()
            self.start.setIcon(QtGui.QIcon(static_path + '/imag/pause.png'))
            self.machine.stop()
        else:
            self.start.setIcon(QtGui.QIcon(static_path + '/imag/play-button.png'))
            self.machine.start()

    def restart_clicked_event(self):
        self.start.setChecked(False)
        self.start_clicked_event()
        self.restart.setEnabled(False)


class PlayerLineInfo(QtWidgets.QWidget):
    def __init__(self, protocols_names, protocols_durations=[], **kwargs):
        super().__init__(**kwargs)

        # init layout
        layer = QtWidgets.QHBoxLayout()
        self.setLayout(layer)

        # status widget
        self.status = QtWidgets.QLabel()
        layer.addWidget(self.status)

        #
        self.n_protocols = len(protocols_names)
        self.protocol_ind = -1
        self.protocols_names = protocols_names
        self.init()

    def init(self):
        self.status.setText('Press play button to start. First protocol is \"{}\"'.format(self.protocols_names[0]))

    def update(self):
        self.protocol_ind += 1
        self.status.setText(
            'Current protocol: \"{}\" \t({}/{} completed)'.format(self.protocols_names[self.protocol_ind],
                                                                  self.protocol_ind, self.n_protocols)
        )

    def finish(self):
        self.status.setText('Protocols sequence is successfully completed!')

    def restart(self):
        self.protocol_ind = -1
        self.init()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, current_protocol, protocols, signals, n_signals=1, parent=None, n_channels=32,
                 max_protocol_n_samples=None,
                 experiment=None, freq=500,
                 plot_raw_flag=True, plot_signals_flag=True, plot_source_space_flag=False, show_subject_window=True,
                 channels_labels=None, photo_rect=False):
        super(MainWindow, self).__init__(parent)

        # Which windows to draw:
        self.plot_source_space_flag = plot_source_space_flag
        self.show_subject_window = show_subject_window

        # status info
        self.status = PlayerLineInfo([p.name for p in protocols], [[p.duration for p in protocols]])

        self.source_freq = freq
        self.experiment = experiment
        self.signals = signals

        # player panel
        self.player_panel = PlayerButtonsWidget(parent=self)
        self.player_panel.restart.clicked.connect(self.restart_experiment)
        self.player_panel.start.clicked.connect(self.update_first_status)
        self._first_time_start_press = True

        # timer label
        self.timer_label = QtWidgets.QLabel('tf')

        # signals viewer
        self.signals_viewer = DerivedSignalViewer(freq, [signal.name for signal in signals])

        # raw data viewer
        self.raw_viewer = RawSignalViewer(freq, channels_labels, notch_filter=True)
        self.n_channels = n_channels
        self.n_samples = 2000

        self.plot_raw_checkbox = QtWidgets.QCheckBox('plot raw')
        self.plot_raw_checkbox.setChecked(plot_raw_flag)
        self.plot_signals_checkbox = QtWidgets.QCheckBox('plot signals')
        self.plot_signals_checkbox.setChecked(plot_signals_flag)
        self.autoscale_raw_chekbox = QtWidgets.QCheckBox('autoscale')
        self.autoscale_raw_chekbox.setChecked(True)

        # topomaper
        # pos = ch_names_to_2d_pos(channels_labels)
        # self.topomaper = TopomapWidget(pos)

        # dc_blocker
        self.dc_blocker = DCBlocker()

        # main window layout
        layout = pg.LayoutWidget(self)
        layout.addWidget(self.signals_viewer, 0, 0, 1, 3)
        layout.addWidget(self.plot_raw_checkbox, 1, 0, 1, 1)
        layout.addWidget(self.plot_signals_checkbox, 1, 2, 1, 1)
        layout.addWidget(self.autoscale_raw_chekbox, 1, 1, 1, 1)
        layout.addWidget(self.raw_viewer, 2, 0, 1, 3)
        layout.addWidget(self.player_panel, 3, 0, 1, 1)
        layout.addWidget(self.timer_label, 3, 1, 1, 1)
        #layout.addWidget(self.topomaper, 3, 2, 1, 1)
        layout.addWidget(self.status, 4, 0, 1, 3)
        layout.layout.setRowStretch(0, 2)
        layout.layout.setRowStretch(2, 2)
        self.setCentralWidget(layout)

        # main window settings
        self.resize(800, 600)
        self.show()

        # subject window
        if show_subject_window:
            self.subject_window = SubjectWindow(self, current_protocol, photo_rect=photo_rect)
            self.subject_window.show()
            self._subject_window_want_to_close = False
        else:
            self.subject_window = None
            self._subject_window_want_to_close = None

        # Source space window
        if plot_source_space_flag:
            source_space_protocol = SourceSpaceRecontructor(signals)
            self.source_space_window = SourceSpaceWindow(self, source_space_protocol)
            self.source_space_window.show()

        # time counter
        self.time_counter = 0
        self.time_counter1 = 0
        self.t0 = time.time()
        self.t = self.t0

    def update_statistics_lines(self):
        pass

    def redraw_signals(self, samples, chunk, samples_counter, n_samples):

        # derived signals
        if self.plot_signals_checkbox.isChecked():
            self.signals_viewer.update(samples)

        # raw signals
        if self.plot_raw_checkbox.isChecked():
            self.raw_viewer.update(chunk)

        # topomaper
        #self.topomaper.set_topomap(np.abs(np.nanmean(self.raw_viewer.y_raw_buffer[-50:], 0)))

        # timer
        if self.time_counter % 10 == 0:
            t_curr = time.time()
            self.timer_label.setText(
                'samples:\t{}\t/{:.0f}\ttime:\t{:.1f}\tfps:\t{:.2f}\tchunk size:\t{}\t '
                    .format(samples_counter, n_samples, t_curr - self.t0, 1. / (t_curr - self.t) * 10, chunk.shape[0]))
            self.t = t_curr
        self.time_counter += 1
        self.time_counter1 += 1

    def restart_experiment(self):
        self.status.restart()
        self.experiment.restart()

    def closeEvent(self, event):
        self._subject_window_want_to_close = True
        if self.show_subject_window and self.subject_window:
            self.subject_window.close()
        if self.plot_source_space_flag and self.source_space_window:
            self.source_space_window.close()
        self.experiment.destroy()
        event.accept()

    def update_first_status(self):
        if self.player_panel.start.isChecked() and self._first_time_start_press:
            self.status.update()
        self._first_time_start_press = False


class SecondaryWindow(QtWidgets.QMainWindow):

    # Must be implemented to return a central widget object in subclasses
    def create_figure(self):
        raise NotImplementedError('create_figure() must be implemented to return a central widget object in subclasses')

    def update_protocol_state(*args, **kwargs):
        raise NotImplementedError('update_protocol_state() must be implemented to update window state')

    def __init__(self, parent, current_protocol, photo_rect=False, **kwargs):
        super().__init__(parent, **kwargs)
        self.resize(500, 500)
        self.current_protocol = current_protocol

        # add central widget
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        widget.setLayout(layout)
        self.figure = self.create_figure()
        layout.setColumnStretch(1, 4)
        layout.addWidget(self.figure, 0, 1, alignment=QtCore.Qt.AlignCenter)

        # add photo sensor rectangle
        if photo_rect:
            self.photo_rect = PhotoRect(white_color=True)
            layout.addWidget(self.photo_rect, 0, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
            layout.addWidget(PhotoRect(), 0, 2, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        else:
            self.photo_rect = None


        self.figure.show_reward(False)
        self.setCentralWidget(widget)

        # background
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(37, 33, 32))
        self.setPalette(p)

        # prepare widget
        self.current_protocol.widget_painter.prepare_widget(self.figure)

    def change_protocol(self, new_protocol):
        self.current_protocol = new_protocol
        self.figure.clear_all()
        self.current_protocol.widget_painter.prepare_widget(self.figure)

    def closeEvent(self, event):
        if self.parent().experiment.is_finished or self.parent()._subject_window_want_to_close:
            event.accept()
        else:
            event.ignore()


class PhotoRect(pg.PlotWidget):
    def __init__(self, size=50, white_color=False, **kwargs):
        super(PhotoRect, self).__init__(**kwargs)
        self.setMaximumWidth(size)
        self.setMaximumHeight(size)
        self.setMinimumWidth(size)
        self.setMinimumHeight(size)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.color = np.ones(3) * 255
        self.setBackgroundBrush(pg.mkBrush(self.color if white_color else (37, 33, 32)))

    def change_color(self, c):
        # c - color in [0,1]
        self.setBackgroundBrush(pg.mkBrush(self.color*(np.tanh(c)*0.5+0.5)))


class SubjectWindow(SecondaryWindow):
    def create_figure(self):
        return ProtocolWidget()

    def update_protocol_state(self, samples, reward, chunk_size=1, is_half_time=False):
        self.current_protocol.update_state(samples=samples, reward=reward, chunk_size=chunk_size,
                                           is_half_time=is_half_time)
        if self.photo_rect is not None:
            self.photo_rect.change_color(samples[self.current_protocol.source_signal_id or 0])

def main():
    print(static_path)
    app = QtWidgets.QApplication(sys.argv)
    widget = PlayerButtonsWidget()
    widget.show()
    sys.exit(app.exec_())
    pass


if __name__ == '__main__':
    main()


class SourceSpaceWindow(SecondaryWindow):
    def create_figure(self):
        return SourceSpaceWidget()

    def update_protocol_state(self, chunk):
        self.current_protocol.update_state(chunk)
