from PyQt4 import QtGui, QtCore
import sys

class ParameterSlider(QtGui.QWidget):
    def __init__(self, minimum=0, maximum=1, interval=0.05, value=0.05):
        super(ParameterSlider, self).__init__()
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)

        # slider
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scaler = 100
        layout.addWidget(slider, 3)
        slider.setRange(minimum * self.scaler, maximum * self.scaler)
        slider.setValue(value * self.scaler)
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        slider.setTickInterval(interval * self.scaler)
        slider.valueChanged.connect(self.set_value_from_slider)
        self.slider = slider

        # line edit
        value_edit = QtGui.QDoubleSpinBox()
        value_edit.setRange(minimum, maximum)
        value_edit.setValue(value)
        value_edit.setSingleStep(interval)
        value_edit.valueChanged.connect(self.set_slider_from_value)
        self.value = value_edit
        layout.addWidget(value_edit, 1)


    def set_value_from_slider(self):
        self.value.setValue(self.slider.value() / self.scaler)

    def set_slider_from_value(self):
        self.slider.setValue(self.value.value() * self.scaler)




if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = ParameterSlider()
    w.show()
    sys.exit(app.exec_())
