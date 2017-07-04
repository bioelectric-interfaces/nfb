from PyQt4 import QtGui

from pynfb.helpers.beep import SingleBeep
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
        self.form_layout.addRow('Reference:', None)
        self.reference = QtGui.QLineEdit(self)
        self.reference.setPlaceholderText('Print channels to exclude (labels)')
        self.reference.textChanged.connect(self.reference_changed_event)
        self.form_layout.addRow('&    Exclude channels:', self.reference)

        self.reference_sub = QtGui.QLineEdit(self)
        self.reference_sub.setPlaceholderText('Print subtractive channel (labels)')
        self.reference_sub.textChanged.connect(self.reference_sub_changed_event)
        self.form_layout.addRow('&    Subtract channel from other:', self.reference_sub)

        # plot raw flag
        self.plot_raw_check = QtGui.QCheckBox()
        self.plot_raw_check.clicked.connect(self.plot_raw_checkbox_event)
        self.form_layout.addRow('&Plot raw:', self.plot_raw_check)
        # plot signals flag
        self.plot_signals_check = QtGui.QCheckBox()
        self.plot_signals_check.clicked.connect(self.plot_signals_checkbox_event)
        self.form_layout.addRow('&Plot signals:', self.plot_signals_check)

        # reward period
        self.reward_period = QtGui.QDoubleSpinBox()
        self.reward_period.setRange(0.05, 10)
        self.reward_period.setSingleStep(0.01)
        self.reward_period.setMaximumWidth(100)
        self.reward_period.valueChanged.connect(self.reward_period_changed_event)
        self.form_layout.addRow('&Reward period [s]:', self.reward_period)

        # beep button
        beep_button = QtGui.QPushButton('Beep')
        beep_button.setMaximumWidth(100)
        beep_button.clicked.connect(lambda : SingleBeep().try_to_play())
        self.form_layout.addRow('&Test beep sound:', beep_button)

        # dc blocker
        self.dc_check = QtGui.QCheckBox()
        self.dc_check.clicked.connect(self.dc_check_event)
        self.form_layout.addRow('&Enable DC Blocker:', self.dc_check)

        # dc blocker
        self.use_expyriment = QtGui.QCheckBox()
        self.use_expyriment.clicked.connect(self.use_expyriment_event)
        self.form_layout.addRow('&Use expyriment toolbox:', self.use_expyriment)

        self.reset()
        # self.stream

    def name_changed_event(self):
        self.params['sExperimentName'] = self.name.text()

    def reference_changed_event(self):
        self.params['sReference'] = self.reference.text()

    def reference_sub_changed_event(self):
        self.params['sReferenceSub'] = self.reference_sub.text()

    def plot_raw_checkbox_event(self):
        self.params['bPlotRaw'] = int(self.plot_raw_check.isChecked())

    def plot_signals_checkbox_event(self):
        self.params['bPlotSignals'] = int(self.plot_signals_check.isChecked())

    def dc_check_event(self):
        self.params['bDC'] = int(self.dc_check.isChecked())

    def use_expyriment_event(self):
        self.params['bUseExpyriment'] = int(self.use_expyriment.isChecked())

    def reward_period_changed_event(self):
        self.params['fRewardPeriodS'] = self.reward_period.value()

    def reset(self):
        self.params = self.parent().params
        self.name.setText(self.params['sExperimentName'])
        self.reference.setText(self.params['sReference'])
        self.reference_sub.setText(self.params['sReferenceSub'])
        self.plot_raw_check.setChecked(self.params['bPlotRaw'])
        self.plot_signals_check.setChecked(self.params['bPlotSignals'])
        self.reward_period.setValue(self.params['fRewardPeriodS'])
        self.dc_check.setChecked(self.params['bDC'])
        self.use_expyriment.setChecked(self.params['bUseExpyriment'])
        self.inlet.reset()