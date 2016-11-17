import pyqtgraph as pg
from PyQt4 import QtGui, QtCore


class CrossButtonsWidget(QtGui.QWidget):
    left_pressed = QtCore.pyqtSignal()
    right_pressed = QtCore.pyqtSignal()
    top_pressed = QtCore.pyqtSignal()
    bottom_pressed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(CrossButtonsWidget, self).__init__(parent)
        left = QtGui.QPushButton('<')
        left.clicked.connect(lambda : self.left_pressed.emit())
        right = QtGui.QPushButton('>')
        right.clicked.connect(lambda: self.right_pressed.emit())
        top = QtGui.QPushButton('^')
        top.clicked.connect(lambda: self.top_pressed.emit())
        bottom = QtGui.QPushButton('v')
        bottom.clicked.connect(lambda: self.bottom_pressed.emit())

        layout = QtGui.QGridLayout(self)
        layout.addWidget(left, 1, 0)
        layout.addWidget(right, 1, 2)
        layout.addWidget(top, 0, 1)
        layout.addWidget(bottom, 2, 1)

a = QtGui.QApplication([])
plot_widget = pg.PlotWidget()

btn = CrossButtonsWidget()

#btn_item = QtGui.QGraphicsProxyWidget()
#btn_item.setWidget(btn)
#plot_widget.addItem(btn_item)
btn.setParent(plot_widget)
btn.left_pressed.connect(lambda: print('LEFT!'))
btn.setGeometry(50, 0, 100, 100)
plot_widget.show()


a.exec_()
