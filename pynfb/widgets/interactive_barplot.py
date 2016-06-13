from PyQt4 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

STYLE = {
    'pen': pg.mkPen(0, 0, 0),
    'pen-hover': pg.mkPen(255, 255, 255, width=1),
    'brush': pg.mkBrush(100, 100, 100),
    'brush-hover': pg.mkBrush(150, 150, 150),
    'brush-selected': pg.mkBrush(200, 200, 200)
}


class ClickableBar(QtGui.QGraphicsRectItem):
    def __init__(self, barplot, x, y, w, h):
        self.barplot = barplot
        self.x = x
        _scale = 10000
        QtGui.QGraphicsRectItem.__init__(self, QtCore.QRectF(x * _scale, y * _scale, w * _scale, h * _scale))
        self.setScale(1 / _scale)
        self.setPen(STYLE['pen'])
        self.setBrush(STYLE['brush'])
        self.setAcceptHoverEvents(True)
        self.is_current_flag = False

    def hoverEnterEvent(self, ev):
        self.savedPen = self.pen()
        self.setPen(STYLE['pen-hover'])
        if not self.is_current_flag:
            self.setBrush(STYLE['brush-hover'])
        ev.ignore()

    def hoverLeaveEvent(self, ev):
        self.setPen(self.savedPen)
        if not self.is_current_flag:
            self.setBrush(STYLE['brush'])
        ev.ignore()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.barplot.set_all_not_current()
            self.set_current(True)
            self.barplot.changed_action()
            ev.accept()
        else:
            ev.ignore()

    def set_current(self, flag):
        if flag:
            self.barplot.current = self
            self.is_current_flag = True
            self.setBrush(STYLE['brush-selected'])
        else:
            self.is_current_flag = False
            self.setBrush(STYLE['brush'])

    def is_current(self):
        return self.is_current_flag


class ClickableBarplot(pg.PlotWidget):
    def __init__(self, parent, x, y, center=False, **kwargs):
        print(parent)
        super(ClickableBarplot, self).__init__(parent=parent, **kwargs)
        self.parent = parent
        self.rectangles = []
        delta = x[1] - x[0]
        for _x, _y in zip(x, y):
            rect = ClickableBar(self, _x - int(center)*delta/2, 0, delta, _y)
            self.addItem(rect)
            self.rectangles.append(rect)
        self.set_all_not_current()

    def set_all_not_current(self):
        self.current = None
        for rectangle in self.rectangles:
            rectangle.set_current(False)
        pass

    def current_index(self):
        return self.rectangles.index(self.current)

    def changed_action(self):
        if self.parent is not None:
            self.parent.select_action()
        else:
            print('Parent is None')


if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = ClickableBarplot(None, np.linspace(0, 1, 50), np.random.uniform(size=50) + np.sin(np.arange(50) / 10) + 1, True)
    widget.show()
    app.exec_()
