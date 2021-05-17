from PyQt5 import QtCore, QtGui, QtWidgets
from ...widgets.parameter_slider import ParameterSlider
from pynfb.signal_processing.decompositions import DEFAULTS as defaults

defaults.update({'bandwidth': 2, 'flanker_bandwidth': 2, 'flanker_margin': 0})

class Sliders(QtWidgets.QWidget):
    def __init__(self):
        super(Sliders, self).__init__()
        h_layout = QtWidgets.QHBoxLayout()
        v_layout = QtWidgets.QVBoxLayout()
        self.setLayout(h_layout)
        h_layout.addLayout(v_layout)
        self.parameters = {}

        # regularizator slider
        self.parameters['regularizator'] = ParameterSlider('Regularization coefficient:', 0, 10, 0.5,
                                                           value=defaults['regularizator'], decimals=3)
        self.parameters['regularizator'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['regularizator'])

        # central bandwidth slider
        self.parameters['bandwidth'] = ParameterSlider('Central bandwidth:', 1, 5, 0.5, value=defaults['bandwidth'])
        self.parameters['bandwidth'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['bandwidth'])

        # flanker bandwidth
        self.parameters['flanker_bandwidth'] = ParameterSlider('Flanker bandwidth:', 1, 5, 0.5, value=defaults['flanker_bandwidth'])
        self.parameters['flanker_bandwidth'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['flanker_bandwidth'])

        # flanker margin
        self.parameters['flanker_margin'] = ParameterSlider('Flanker margin:', 0, 2, 0.5, value=defaults['flanker_margin'])
        self.parameters['flanker_margin'].slider.valueChanged.connect(lambda: self.revert_button.setEnabled(True))
        v_layout.addWidget(self.parameters['flanker_margin'])

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
        for key in defaults.keys():
            self.parameters[key].setValue(defaults[key])
        self.revert_button.setEnabled(False)

    def getValues(self):
        values = dict([(key, param.getValue()) for key, param in self.parameters.items()])
        return values


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = Sliders()
    widget.show()
    app.exec_()
