from PyQt5 import QtCore, QtGui, QtWidgets
from ...widgets.parameter_slider import ParameterSlider


class Sliders(QtWidgets.QWidget):
    def __init__(self, sample_freq, reg_coef=True, stimulus_split=True, band=None):
        super(Sliders, self).__init__()
        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()
        self.setLayout(h_layout)
        h_layout.addLayout(v_layout)
        self.parameters = {}
        self.sample_freq = sample_freq

        self.defaults = {'bandpass_low': 3 if band is None else band[0],
                        'regularizator': 0.05,
                        'bandpass_high': 45 if band is None else band[1],
                        'prestim_interval': 500,
                        'poststim_interval': 500}


        # regularizator slider
        self.parameters['regularizator'] = ParameterSlider('Regularization coefficient:', 0, 10, 0.5,
                                                           value=self.defaults['regularizator'])
        self.parameters['regularizator'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['regularizator'])
        if not reg_coef:
            self.parameters['regularizator'].hide()


        # prestim/poststim intervals:
        self.parameters['prestim_interval'] = ParameterSlider('PRE-stimulus interval:', 0, 2000, 100,
                                                           value=self.defaults['prestim_interval'], integer=True)
        self.parameters['prestim_interval'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['prestim_interval'])
        self.parameters['poststim_interval'] = ParameterSlider('POST-stimulus interval:', 0, 2000, 100,
                                                           value=self.defaults['poststim_interval'], integer=True)
        self.parameters['poststim_interval'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['poststim_interval'])


        if not stimulus_split:
            self.parameters['prestim_interval'].hide()
            self.parameters['poststim_interval'].hide()

        # central bandpass_low slider
        self.parameters['bandpass_low'] = ParameterSlider('Bandpass low:', 0, sample_freq/2, 1,
                                                          value=self.defaults['bandpass_low'])
        self.parameters['bandpass_low'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['bandpass_low'])

        # flanker bandpass_low
        self.parameters['bandpass_high'] = ParameterSlider('Bandpass high:', 0, sample_freq/2, 1,
                                                           value=self.defaults['bandpass_high'])
        self.parameters['bandpass_high'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['bandpass_high'])

        button_layout = QtWidgets.QVBoxLayout()
        h_layout.addLayout(button_layout)
        # apply button
        self.apply_button = QtWidgets.QPushButton('Apply')
        button_layout.addWidget(self.apply_button)

        # revert button
        self.revert_button = QtWidgets.QPushButton('Restore\ndefaults')
        self.revert_button.setEnabled(False)
        self.revert_button.clicked.connect(self.restore_defaults)
        button_layout.addWidget(self.revert_button)

    def restore_defaults(self):
        for key in self.defaults.keys():
            self.parameters[key].setValue(self.defaults[key])
        self.revert_button.setEnabled(False)

    def set_default_band(self, low, high):
        self.defaults['bandpass_low'] = low
        self.defaults['bandpass_high'] = high
        self.restore_defaults()

    def getValues(self):
        values = dict([(key, param.getValue()) for key, param in self.parameters.items()])
        return values


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = Sliders()
    widget.show()
    app.exec_()
