import pyqtgraph as pg
from PyQt4 import QtGui, QtCore


class CrossButtonsWidget(QtGui.QWidget):
    names = ['left', 'right', 'up', 'down']
    symbols = ['<', '>', '^', 'v']
    positions = [(1, 0), (1, 2), (0, 1), (2, 1)]

    def __init__(self, parent=None):
        super(CrossButtonsWidget, self).__init__(parent)
        buttons = dict([(name, QtGui.QPushButton(symbol)) for name, symbol in zip(self.names, self.symbols)])
        layout = QtGui.QGridLayout(self)
        self.clicked_dict = {}
        for name, pos in zip(self.names, self.positions):
            layout.addWidget(buttons[name], *pos)
            self.clicked_dict[name] = buttons[name].clicked



if __name__ == '__main__':
    a = QtGui.QApplication([])
    plot_widget = pg.PlotWidget()

    btn = CrossButtonsWidget()

    #btn_item = QtGui.QGraphicsProxyWidget()
    #btn_item.setWidget(btn)
    #plot_widget.addItem(btn_item)
    btn.setParent(plot_widget)
    btn.clicked_dict['left'].connect(lambda: print('LEFT!'))
    btn.clicked_dict['down'].connect(lambda: print('DOWN!'))
    btn.clicked_dict['up'].connect(lambda: print('UP!'))
    btn.clicked_dict['right'].connect(lambda: print('R!'))
    btn.setGeometry(50, 0, 100, 100)
    plot_widget.show()
    a.exec_()
