from PyQt4 import QtGui, QtCore
import sys

class ParameterSlider(QtGui.QWidget):
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, label, minimum=0, maximum=1, interval=0.05, value=0.05, units='', integer=False):
        super(ParameterSlider, self).__init__()
        layout = QtGui.QHBoxLayout()
        layout.setMargin(0)
        self.setLayout(layout)

        # label
        layout.addWidget(QtGui.QLabel(label), 1)

        # slider
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scaler = 5
        layout.addWidget(slider, 3)
        slider.setRange(minimum * self.scaler, maximum * self.scaler)
        slider.setValue(value * self.scaler)
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        slider.setTickInterval(interval * self.scaler)
        slider.valueChanged.connect(self.set_value_from_slider)
        self.slider = slider

        # line edit
        value_edit = QtGui.QDoubleSpinBox() if not integer else QtGui.QSpinBox()
        value_edit.setRange(minimum, maximum)
        value_edit.setValue(value)
        value_edit.setSingleStep(interval)
        value_edit.valueChanged.connect(self.set_slider_from_value)
        value_edit.valueChanged.connect(lambda: self.valueChanged.emit())
        self.value = value_edit
        layout.addWidget(value_edit, 1)

        #units
        if units:
            layout.addWidget(QtGui.QLabel(units))

    def set_value_from_slider(self):
        self.value.setValue(self.slider.value() / self.scaler)

    def set_slider_from_value(self):
        self.slider.setValue(self.value.value() * self.scaler)

    def setValue(self, p_float):
        self.value.setValue(p_float)

    def getValue(self):
        return self.value.value()




if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = ParameterSlider('Regularization')
    w.show()
    sys.exit(app.exec_())
