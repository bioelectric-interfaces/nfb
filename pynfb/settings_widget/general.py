from PyQt4 import QtGui
from .inlet import InletSettingsWidget

class GeneralSettingsWidget(QtGui.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params
        self.form_layout = QtGui.QFormLayout(self)
        self.setLayout(self.form_layout)

        # name
        self.name = QtGui.QLineEdit(self)
        self.name.textChanged.connect(self.name_changed_event)
        self.form_layout.addRow('&Name:', self.name)

        # composite montage
        # self.montage = QtGui.QLineEdit(self)
        # self.montage.setPlaceholderText('Print path to file')
        # self.form_layout.addRow('&Composite\nmontage:', self.montage)

        # inlet
        self.inlet = InletSettingsWidget(parent=self)
        self.form_layout.addRow('&Inlet:', self.inlet)

        # reference
        self.reference = QtGui.QLineEdit(self)
        self.reference.setPlaceholderText('Print reference (names or numbers)')
        self.reference.textChanged.connect(self.reference_changed_event)
        self.form_layout.addRow('&Reference:', self.reference)

        # plot raw flag
        self.plot_raw_check = QtGui.QCheckBox()
        self.plot_raw_check.clicked.connect(self.plot_raw_checkbox_event)
        self.form_layout.addRow('&Plot raw:', self.plot_raw_check)
        # plot signals flag
        self.plot_signals_check = QtGui.QCheckBox()
        self.plot_signals_check.clicked.connect(self.plot_signals_checkbox_event)
        self.form_layout.addRow('&Plot signals:', self.plot_signals_check)

        self.reset()
        # self.stream

    def name_changed_event(self):
        self.params['sExperimentName'] = self.name.text()

    def reference_changed_event(self):
        self.params['sReference'] = self.reference.text()

    def plot_raw_checkbox_event(self):
        self.params['bPlotRaw'] = int(self.plot_raw_check.isChecked())

    def plot_signals_checkbox_event(self):
        self.params['bPlotSignals'] = int(self.plot_signals_check.isChecked())

    def reset(self):
        self.params = self.parent().params
        self.name.setText(self.params['sExperimentName'])
        self.reference.setText(self.params['sReference'])
        self.plot_raw_check.setChecked(self.params['bPlotRaw'])
        self.plot_signals_check.setChecked(self.params['bPlotSignals'])
        self.inlet.reset()