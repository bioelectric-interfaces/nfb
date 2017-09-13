import os
import sys

from PyQt4 import QtGui, QtCore

from pynfb.experiment import Experiment
from pynfb.io.xml_ import xml_file_to_params
from pynfb.settings_widget.general import GeneralSettingsWidget
from pynfb.settings_widget.inlet import InletSettingsWidget
from pynfb.settings_widget.protocol_sequence import ProtocolSequenceSettingsWidget
from pynfb.settings_widget.protocols import ProtocolsSettingsWidget, FileSelectorLine
from pynfb.settings_widget.signals import SignalsSettingsWidget
from pynfb.settings_widget.composite_signals import CompositeSignalsSettingsWidget
from pynfb.settings_widget.protocols_group import ProtocolGroupsSettingsWidget

static_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static')


class SettingsWidget(QtGui.QWidget):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        v_layout = QtGui.QVBoxLayout()
        layout = QtGui.QHBoxLayout()
        self.params = xml_file_to_params('sourcespace.xml')
        self.general_settings = GeneralSettingsWidget(parent=self)
        v_layout.addWidget(self.general_settings)
        v_layout.addLayout(layout)
        self.protocols_list = ProtocolsSettingsWidget(parent=self)
        self.signals_list = SignalsSettingsWidget(parent=self)
        self.composite_signals_list = CompositeSignalsSettingsWidget(parent=self)
        self.protocol_groups_list = ProtocolGroupsSettingsWidget(parent=self)
        self.protocols_sequence_list = ProtocolSequenceSettingsWidget(parent=self)
        # layout.addWidget(self.general_settings)
        layout.addWidget(self.signals_list)
        layout.addWidget(self.composite_signals_list)
        layout.addWidget(self.protocols_list)
        layout.addWidget(self.protocol_groups_list)
        layout.addWidget(self.protocols_sequence_list)
        start_button = QtGui.QPushButton('Start')
        start_button.setIcon(QtGui.QIcon(static_path + '/imag/power-button.png'))
        start_button.setMinimumHeight(50)
        start_button.setMinimumWidth(300)
        start_button.clicked.connect(self.onClicked)
        name_layout = QtGui.QHBoxLayout()
        v_layout.addWidget(start_button, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(v_layout)

    def reset_parameters(self):
        self.signals_list.reset_items()
        self.composite_signals_list.reset_items()
        self.protocols_list.reset_items()
        self.protocols_sequence_list.reset_items()
        self.protocol_groups_list.reset_items()
        self.general_settings.reset()
        # self.params['sExperimentName'] = self.experiment_name.text()

    def onClicked(self):
        self.experiment = Experiment(self.app, self.params)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = FileSelectorLine()
    window.show()
    sys.exit(app.exec_())
