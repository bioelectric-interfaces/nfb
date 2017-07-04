import os
import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSignal
from pynfb.protocols.widgets import *
from numpy import isnan
from expyriment import control, design, misc
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.widgets.signals_painter import RawViewer
from pynfb.widgets.topography import TopomapWidget
from pynfb.helpers.dc_blocker import DCBlocker

pg.setConfigOptions(antialias=True)

static_path = full_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../static')


class LSLPlotDataItem(pg.PlotDataItem):
    def getData(self):
        x, y = super(LSLPlotDataItem, self).getData()
        if self.opts['fftMode']:
            return x, y / (max(y) - min(y) + 1e-20) * 0.75
        return x, y


class PlayerButtonsWidget(QtGui.QWidget):
    start_clicked = pyqtSignal()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init buttons
        self.start = QtGui.QPushButton('')
        self.restart = QtGui.QPushButton('')
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
        layer = QtGui.QHBoxLayout()
        self.setLayout(layer)
        layer.addWidget(self.start)
        layer.addWidget(self.restart)
        self.setMaximumWidth(200)
        self.setMinimumWidth(100)

        styles = ["background-color: #{}".format(color) for color in ["FFFFFF", "F8FFFB", "F2FFF8", "ECFFF4", "E6FFF1",
                                                                      "DFFFED", "D9FFEA", "D3FFE6", "CDFFE3", "C7FFE0"]]
        styles += styles[::-1]

        # animation doesn't work for strings but provides an appropriate delay
        animation = QtCore.QPropertyAnimation(self.start, 'styleSheet')
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


class PlayerLineInfo(QtGui.QWidget):
    def __init__(self, protocols_names, protocols_durations=[], **kwargs):
        super().__init__(**kwargs)

        # init layout
        layer = QtGui.QHBoxLayout()
        self.setLayout(layer)

        # status widget
        self.status = QtGui.QLabel()
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


class MainWindow(QtGui.QMainWindow):
    def __init__(self, current_protocol, protocols, signals, n_signals=1, parent=None, n_channels=32,
                 max_protocol_n_samples=None,
                 experiment=None, freq=500, plot_raw_flag=True, plot_signals_flag=True, channels_labels=None,
                 subject_backend_expyriment=False):
        super(MainWindow, self).__init__(parent)

        # status info
        self.status = PlayerLineInfo([p.name for p in protocols], [[p.duration for p in protocols]])

        self.source_freq = freq
        self.experiment = experiment
        self.signals = signals

        # player panel
        self.player_panel = PlayerButtonsWidget(parent=self)
        self.player_panel.restart.clicked.connect(self.restart_experiment)
        for signal in signals:
            self.player_panel.start.clicked.connect(signal.reset_statistic_acc)
        self.player_panel.start.clicked.connect(self.update_first_status)
        self._first_time_start_press = True

        # timer label
        self.timer_label = QtGui.QLabel('tf')

        # signals viewer
        self.signals_viewer = RawViewer(freq, channels_labels=[signal.name for signal in signals], overlap=True)

        # raw data viewer
        self.raw_viewer = RawViewer(freq, channels_labels)
        self.n_channels = n_channels
        self.n_samples = 2000

        self.plot_raw_checkbox = QtGui.QCheckBox('plot raw')
        self.plot_raw_checkbox.setChecked(plot_raw_flag)
        self.plot_signals_checkbox = QtGui.QCheckBox('plot signals')
        self.plot_signals_checkbox.setChecked(plot_signals_flag)
        self.autoscale_raw_chekbox = QtGui.QCheckBox('autoscale')
        self.autoscale_raw_chekbox.setChecked(True)

        # topomaper
        pos = ch_names_to_2d_pos(channels_labels)
        self.topomaper = TopomapWidget(pos)

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
        layout.addWidget(self.topomaper, 3, 2, 1, 1)
        layout.addWidget(self.status, 4, 0, 1, 3)
        layout.layout.setRowStretch(0, 2)
        layout.layout.setRowStretch(2, 2)
        self.setCentralWidget(layout)

        # main window settings
        self.resize(800, 600)
        self.show()

        # subject window
        if not subject_backend_expyriment:
            self.subject_window = SubjectWindow(self, current_protocol)
            self.subject_window.show()
        else:
            self.subject_window = ExpyrimentSubjectWindow(self, current_protocol)
        self._subject_window_want_to_close = False

        # time counter
        self.time_counter = 0
        self.time_counter1 = 0
        self.t0 = time.time()
        self.t = self.t0

    def update_statistics_lines(self):
        pass

    def redraw_signals(self, samples, chunk, samples_counter):

        # derived signals
        if self.plot_signals_checkbox.isChecked():
            if self.time_counter1 < 10:
                self.signals_viewer.update_std(np.dot(np.ones((chunk.shape[0], 1)), samples[None, :]))
                self.signals_viewer.update_levels()
            else:
                self.signals_viewer.set_chunk(np.dot(np.ones((chunk.shape[0], 1)), samples[None, :]))

        # raw signals
        if self.plot_raw_checkbox.isChecked():

            if self.time_counter < 10:
                self.raw_viewer.update_std(chunk)
            else:
                self.raw_viewer.set_chunk(chunk)

        # topomaper
        self.topomaper.set_topomap(np.abs(self.raw_viewer.raw_buffer[-50:]).mean(0))

        # timer
        if self.time_counter % 10 == 0:
            t_curr = time.time()
            self.timer_label.setText(
                'samples:\t{}\ttime:\t{:.1f}\tfps:\t{:.2f}\tchunk size:\t{}\t '
                    .format(samples_counter, t_curr - self.t0, 1. / (t_curr - self.t) * 10, chunk.shape[0]))
            self.t = t_curr
        self.time_counter += 1
        self.time_counter1 += 1

    def restart_experiment(self):
        self.status.restart()
        self.experiment.restart()

    def closeEvent(self, event):
        self._subject_window_want_to_close = True
        self.subject_window.close()
        self.experiment.destroy()
        event.accept()

    def update_first_status(self):
        if self.player_panel.start.isChecked() and self._first_time_start_press:
            self.status.update()
        self._first_time_start_press = False


class SubjectWindow(QtGui.QMainWindow):
    def __init__(self, parent, current_protocol, **kwargs):
        super(SubjectWindow, self).__init__(parent, **kwargs)
        self.resize(500, 500)
        self.current_protocol = current_protocol

        # add central widget
        widget = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()
        widget.setLayout(layout)
        self.figure = ProtocolWidget()
        layout.addWidget(self.figure)
        self.figure.show_reward(False)
        self.setCentralWidget(widget)

        # background
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(37, 33, 32))
        self.setPalette(p)

        # prepare widget
        self.current_protocol.widget_painter.prepare_widget(self.figure)

    def update_protocol_state(self, samples, chunk_size=1, is_half_time=False):
        self.current_protocol.update_state(samples, chunk_size=chunk_size, is_half_time=is_half_time)
        pass

    def change_protocol(self, new_protocol):
        self.current_protocol = new_protocol
        self.figure.clear_all()
        self.current_protocol.widget_painter.prepare_widget(self.figure)

    def closeEvent(self, event):
        if self.parent().experiment.is_finished or self.parent()._subject_window_want_to_close:
            event.accept()
        else:
            event.ignore()


class CustomExperiment(design.Experiment):
    def clear_all(self):
        pass

    def update_reward(self, reward):
        pass

    def show_reward(self, flag):
        pass

class ExpyrimentSubjectWindow:
    def __init__(self, parent, current_protocol, **kwargs):
        control.defaults.initialize_delay = 0
        control.defaults.window_mode = True
        self.exp = CustomExperiment(background_colour=(0, 0, 0))

        control.initialize(self.exp)
        self.figure = ProtocolWidget()
        self.current_protocol = current_protocol

        # prepare widget
        self.current_protocol.widget_painter.prepare_widget(self.exp)

    def update_protocol_state(self, samples, chunk_size=1, is_half_time=False):
        self.current_protocol.update_state(samples, chunk_size=chunk_size, is_half_time=is_half_time)

    def change_protocol(self, new_protocol):
        self.current_protocol = new_protocol
        self.current_protocol.widget_painter.prepare_widget(self.exp)

    def close(self):
        control.end()

def main():
    print(static_path)
    app = QtGui.QApplication(sys.argv)
    widget = PlayerButtonsWidget()
    widget.show()
    sys.exit(app.exec_())
    pass


if __name__ == '__main__':
    main()
