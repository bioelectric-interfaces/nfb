from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from ...widgets.parameter_slider import ParameterSlider
from time import time




class BandSelectorWidget(QtWidgets.QDialog):
    def __init__(self, x, y):
        super(BandSelectorWidget, self).__init__()

        # parameters
        self.width = 3
        self.x = 20
        self.x_max = 30.
        self.gran = x[1] - x[0]

        graphics_widget = pg.GraphicsWidget()

        self.middle = pg.LinearRegionItem([self.x, self.x], pg.LinearRegionItem.Vertical)
        self.middle.setBounds([0+self.width/2, self.x_max-self.width/2])
        self.middle.sigRegionChanged.connect(self.regionChanged)
        self.region = pg.LinearRegionItem([self.x - self.width/2, self.x + self.width/2], pg.LinearRegionItem.Vertical, movable=False)
        self.region.setBounds((0, self.x_max))
        self.region.sigRegionChanged.connect(self.regionChanged)

        view_box = pg.ViewBox(parent=graphics_widget, enableMouse=False, enableMenu=False)
        #view_box.setAspectLocked()
        view_box.setYRange(0, 1)
        #view_box.enableAutoRange(view_box.XYAxes)

        axis = pg.AxisItem('bottom', linkView=view_box, parent=graphics_widget)
        plot = pg.PlotDataItem()
        plot.setData(x, y/(max(y[x<self.x_max]) or 1))
        view_box.setXRange(0, self.x_max)

        view_box.addItem(plot)
        view_box.addItem(self.region)
        view_box.addItem(self.middle)
        self.view_box = view_box

        # setup grid layout
        grid_layout = QtWidgets.QGraphicsGridLayout(graphics_widget)
        grid_layout.addItem(axis, 1, 0)
        grid_layout.addItem(view_box, 0, 0)

        # view
        view = pg.GraphicsView()
        view.setCentralItem(graphics_widget)
        self.width_slider = ParameterSlider('Band width:', 1, 10, interval=0.1, value=self.width, units='Hz')
        self.width_slider.valueChanged.connect(self.changeWidth)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(view)
        main_layout.addWidget(self.width_slider)
        self.band = None
        btn = QtWidgets.QPushButton('Select')
        btn.clicked.connect(self.select_band)
        btn.setMaximumWidth(200)


        self.band_str = QtWidgets.QLabel('Band:\t{}\t-\t{} Hz'.format(*self.region.getRegion()))
        main_layout.addWidget(self.band_str)
        main_layout.addWidget(btn)

    def changeWidth(self):
        self.width = self.width_slider.value.value()
        self.region.setRegion((self.x - self.width / 2, self.x + self.width / 2))
        self.middle.setBounds((self.width / 2, self.x_max - self.width / 2))
        self.region.setBounds((0, self.x_max))
        self.band_changed()

    def band_changed(self):
        self.band_str.setText('Band:\t{:.1f}\t-\t{:.1f} Hz'.format(*self.region.getRegion()))


    def regionChanged(self):
        try:
            x1, x2 = self.middle.getRegion()
        except RecursionError:
            return
        self.x += x1 - self.x + x2 - self.x
        self.middle.setRegion((self.x, self.x))
        self.region.setRegion((self.x-self.width/2, self.x+self.width/2))
        self.band_changed()

    def select_band(self):
        self.band = self.region.getRegion()
        self.band = (round(self.band[0], 1), round(self.band[1], 1))
        self.close()

    @staticmethod
    def select(x, y):
        w = BandSelectorWidget(x, y)
        w.exec_()
        return w.band




if __name__ == '__main__':
    a = QtWidgets.QApplication([])
    x = np.random.randint(0, 10, size=1000) + np.arange(1000)
    print(BandSelectorWidget.select(np.linspace(0, 250, 1000), x))
    a.exec_()
