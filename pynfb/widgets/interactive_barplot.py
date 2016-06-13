from PyQt4 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

STYLE = {
    'pen': pg.mkPen(0,0,0),
    'pen-hover': pg.mkPen(255, 255, 255, width=1),
    'brush': pg.mkBrush(100, 100, 100),
    'brush-hover': pg.mkBrush(150, 150, 150),
    'brush-selected': pg.mkBrush(200, 200, 200)
}


class MovableRect(QtGui.QGraphicsRectItem):
    def __init__(self, parent, x, y, w, h):
        self.parent = parent
        self.x = x
        _scale = 10000
        QtGui.QGraphicsRectItem.__init__(self, QtCore.QRectF(x*_scale, y*_scale, w*_scale, h*_scale))
        self.setScale(1 / _scale)
        self.setPen(STYLE['pen'])
        self.setBrush(STYLE['brush'])
        self.setAcceptHoverEvents(True)
        self.setBoundingRegionGranularity(1)
        self.is_selected = False
    def hoverEnterEvent(self, ev):
        self.savedPen = self.pen()
        self.setPen(STYLE['pen-hover'])
        if not self.is_selected:
            self.setBrush(STYLE['brush-hover'])
        ev.ignore()
    def hoverLeaveEvent(self, ev):
        self.setPen(self.savedPen)
        if not self.is_selected:
            self.setBrush(STYLE['brush'])
        ev.ignore()
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            print(self.x)
            self.parent.set_all_not_selected()
            self.set_selected(True)
            ev.accept()
            self.pressDelta = self.mapToParent(ev.pos()) - self.pos()
        else:
            ev.ignore()
    def set_selected(self, flag):
        if flag:
            self.is_selected = True
            self.setBrush(STYLE['brush-selected'])
        else:
            self.is_selected = False
            self.setBrush(STYLE['brush'])





class ClickableBarplot(pg.PlotWidget):
    def __init__(self, **kwargs):
        super(ClickableBarplot, self).__init__(**kwargs)
        self.rectangles = []
        for x, y in zip(np.arange(50), np.random.uniform(size=50)+np.sin(np.arange(50)/10)+1):
            rect = MovableRect(self, x, 0, 1, y)
            self.addItem(rect)
            self.rectangles.append(rect)

    def set_all_not_selected(self):
        for rectangle in self.rectangles:
            rectangle.set_selected(False)
        pass

if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = ClickableBarplot()
    widget.show()
    app.exec_()