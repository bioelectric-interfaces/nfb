from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg

STYLE = {
    'pen': pg.mkPen(0, 0, 0),
    'pen-hover': pg.mkPen(255, 255, 255, width=1),
    'brush': pg.mkBrush(100, 100, 100),
    'brush-hover': pg.mkBrush(150, 150, 150),
    'brush-selected': pg.mkBrush(200, 200, 200),
    'underline-central': pg.mkPen(176, 35, 48, width=3),
    'underline-flanker': pg.mkPen(51, 152, 188, width=3)
}
STYLE['underline-central'].setCapStyle(QtCore.Qt.FlatCap)
STYLE['underline-flanker'].setCapStyle(QtCore.Qt.FlatCap)


class ClickableBar(QtWidgets.QGraphicsRectItem):
    def __init__(self, barplot, x, y, w, h):
        self.barplot = barplot
        self.x = x
        _scale = 10000
        QtWidgets.QGraphicsRectItem.__init__(self, QtCore.QRectF(x * _scale, y * _scale, w * _scale, h * _scale))
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

    def set_height(self, h):
        rect = self.rect()
        self.setRect(rect.x(), rect.y(), rect.width(), h)


class ClickableBarplot(pg.PlotWidget):
    changed = QtCore.pyqtSignal()
    def __init__(self, parent, xlabel='Hz', **kwargs):
        super(ClickableBarplot, self).__init__(parent=parent, **kwargs)
        self.parent = parent
        self.rectangles = []
        self.underlines = []
        self.bin_level = None
        self.ticks = []
        self.current = None
        self.getPlotItem().getAxis('bottom').setLabel(xlabel)

    def plot(self, x, y):
        self.clear()
        self.rectangles = []
        for _x, _y in zip(x, y):
            rect = ClickableBar(self, _x, 0, x[1] - x[0], _y)
            self.addItem(rect)
            self.rectangles.append(rect)
        self.set_all_not_current()
        self.current = 0
        self.set_current(0)
        self.setYRange(-0.05*max(y), max(y))

    def set_all_not_current(self):
        self.current = None
        for rectangle in self.rectangles:
            rectangle.set_current(False)
        pass

    def current_index(self):
        return self.rectangles.index(self.current) if self.current is not None else None

    def current_x(self):
        return self.current.x if self.current is not None else 0

    def set_current(self, ind):
        self.set_all_not_current()
        self.rectangles[ind].set_current(True)
        self.changed.emit()

    def set_current_by_value(self, val):
        x = np.array([rect.x for rect in self.rectangles])
        ind = np.abs(x - val).argmin()
        self.set_current(ind)

    def changed_action(self):
        self.changed.emit()
        if self.parent is not None:
            self.parent.select_action()
        else:
            print('Parent is None')

    def underline(self, x1=5, x2=8, style='central'):
        y = -0.02
        item = QtWidgets.QGraphicsLineItem(QtCore.QLineF(x1, y, x2, y))
        item.setPen(STYLE['underline-'+style])
        self.addItem(item)
        self.underlines.append(item)

    def add_xtick(self, val):
        item = pg.TextItem(str(val), anchor=(0.5, 0))
        item.setX(val)
        item.setY(0)
        self.addItem(item)
        self.ticks.append(item)

    def clear_underlines_and_ticks(self):
        for item in self.underlines + self.ticks:
            self.removeItem(item)

    def update_bin_level(self, delta=1, y=1):
        if self.bin_level is not None:
            self.removeItem(self.bin_level)
        x1 = self.current_x()
        x2 = x1 + delta
        item = QtWidgets.QGraphicsLineItem(QtCore.QLineF(x1, y, x2, y))
        item.setPen(STYLE['underline-central'])
        self.bin_level = item
        self.addItem(item)

    def reset_y(self, y):
        self.y = y
        for y_, rect in zip(y, self.rectangles):
            rect.set_height(y_)



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = ClickableBarplot(None, np.linspace(0, 1, 50), np.random.uniform(size=50) + np.sin(np.arange(50) / 10) + 1, True)
    widget.show()
    app.exec_()
