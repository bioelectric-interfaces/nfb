from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from pynfb.helpers.beep import SingleBeep
from .inlet import InletSettingsWidget, EventsInletSettingsWidget


class BandWidget(QtWidgets.QWidget):
    bandChanged = pyqtSignal()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bandpass_layout = QtWidgets.QHBoxLayout(self)
        bandpass_layout.setContentsMargins(0, 0, 0, 0)
        self.band = [QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()]
        for w, name in zip(self.band, ['low:', 'high:']):
            w.setRange(0, 250)
            w.setValue(0)
            w.setMaximumWidth(200)
            label = QtWidgets.QLabel(name)
            label.setMaximumWidth(75)
            bandpass_layout.addWidget(label)
            bandpass_layout.addWidget(w)
            w.valueChanged.connect(self.bandChanged.emit)
        self.setMaximumWidth(600)

    def set_band(self, band_str):
        for j, w_str in enumerate(band_str.split(' ')):
            self.band[j].setValue(float(w_str) if w_str != 'None' else 0)

    def get_band(self):
        band = [None, None]
        for j in range(2):
            value = self.band[j].value()
            band[j] = str(value) if value else 'None'
        return ' '.join(band)


class GeneralSettingsWidget(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = self.parent().params
        self.form_layout = QtWidgets.QFormLayout(self)
        self.setLayout(self.form_layout)

        # name
        self.name = QtWidgets.QLineEdit(self)
        self.name.textChanged.connect(self.name_changed_event)
        self.form_layout.addRow('&Name:', self.name)

        # composite montage
        # self.montage = QtGui.QLineEdit(self)
        # self.montage.setPlaceholderText('Print path to file')
        # self.form_layout.addRow('&Composite\nmontage:', self.montage)

        # inlet
        self.inlet = InletSettingsWidget(parent=self)
        self.form_layout.addRow('&Inlet:', self.inlet)

        # events inlet
        self.events_inlet = EventsInletSettingsWidget(parent=self)
        self.form_layout.addRow('&Events inlet:', self.events_inlet)

        # reference
        self.form_layout.addRow('Reference:', None)
        self.reference = QtWidgets.QLineEdit(self)
        self.reference.setPlaceholderText('Print channels to exclude (labels)')
        self.reference.textChanged.connect(self.reference_changed_event)
        self.form_layout.addRow('&    Exclude channels:', self.reference)

        self.reference_sub = QtWidgets.QLineEdit(self)
        self.reference_sub.setPlaceholderText('Print subtractive channel (labels)')
        self.reference_sub.textChanged.connect(self.reference_sub_changed_event)
        self.form_layout.addRow('&    Subtract channel from other:', self.reference_sub)

        # plot raw flag
        self.plot_raw_check = QtWidgets.QCheckBox()
        self.plot_raw_check.clicked.connect(self.plot_raw_checkbox_event)
        self.form_layout.addRow('&Plot raw:', self.plot_raw_check)
        # plot signals flag
        self.plot_signals_check = QtWidgets.QCheckBox()
        self.plot_signals_check.clicked.connect(self.plot_signals_checkbox_event)
        self.form_layout.addRow('&Plot signals:', self.plot_signals_check)

        # plot source space flag
        self.plot_source_space_check = QtWidgets.QCheckBox()
        self.plot_source_space_check.clicked.connect(self.plot_source_space_checkbox_event)
        self.form_layout.addRow('&Plot source space:', self.plot_source_space_check)

        # show subject window
        self.show_subject_window_check = QtWidgets.QCheckBox()
        self.show_subject_window_check.clicked.connect(self.show_subject_window_checkbox_event)
        self.form_layout.addRow('&Show subject window:', self.show_subject_window_check)

        # reward period
        self.reward_period = QtWidgets.QDoubleSpinBox()
        self.reward_period.setRange(0.05, 10)
        self.reward_period.setSingleStep(0.01)
        self.reward_period.setMaximumWidth(100)
        self.reward_period.valueChanged.connect(self.reward_period_changed_event)
        self.form_layout.addRow('&Reward period [s]:', self.reward_period)

        # beep button
        beep_button = QtWidgets.QPushButton('Beep')
        beep_button.setMaximumWidth(100)
        beep_button.clicked.connect(lambda : SingleBeep().try_to_play())
        self.form_layout.addRow('&Test beep sound:', beep_button)

        # dc blocker
        self.dc_check = QtWidgets.QCheckBox()
        self.dc_check.clicked.connect(self.dc_check_event)
        self.form_layout.addRow('&Enable DC Blocker:', self.dc_check)

        # photo sensor show checkbox
        self.show_photo_rect = QtWidgets.QCheckBox()
        self.show_photo_rect.clicked.connect(self.show_photo_rect_event)
        self.form_layout.addRow('&Show photo-sensor rect.:', self.show_photo_rect)

        # pre-filtering band:
        self.prefilter_band = BandWidget()
        self.prefilter_band.bandChanged.connect(self.band_changed_event)
        self.form_layout.addRow('&Pre-filtering band:', self.prefilter_band)

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

    def plot_source_space_checkbox_event(self):
        self.params['bPlotSourceSpace'] = int(self.plot_source_space_check.isChecked())

    def show_subject_window_checkbox_event(self):
        self.params['bShowSubjectWindow'] = int(self.show_subject_window_check.isChecked())

    def dc_check_event(self):
        self.params['bDC'] = int(self.dc_check.isChecked())

    def show_photo_rect_event(self):
        self.params['bShowPhotoRectangle'] = int(self.show_photo_rect.isChecked())

    def band_changed_event(self):
        self.params['sPrefilterBand'] = self.prefilter_band.get_band()


    def reward_period_changed_event(self):
        self.params['fRewardPeriodS'] = self.reward_period.value()

    def reset(self):
        self.params = self.parent().params
        self.name.setText(self.params['sExperimentName'])
        self.reference.setText(self.params['sReference'])
        self.reference_sub.setText(self.params['sReferenceSub'])
        self.plot_raw_check.setChecked(self.params['bPlotRaw'])
        self.plot_signals_check.setChecked(self.params['bPlotSignals'])
        self.plot_source_space_check.setChecked(self.params['bPlotSourceSpace'])
        self.show_subject_window_check.setChecked(self.params['bShowSubjectWindow'])
        self.reward_period.setValue(self.params['fRewardPeriodS'])
        self.dc_check.setChecked(self.params['bDC'])
        self.show_photo_rect.setChecked(self.params['bShowPhotoRectangle'])
        self.prefilter_band.set_band(self.params['sPrefilterBand'])
        self.inlet.reset()
        self.events_inlet.reset()
