import pyqtgraph as pg
import numpy as np

class ProtocolWidget(pg.PlotWidget):
    def __init__(self, **kwargs):
        super(ProtocolWidget, self).__init__(**kwargs)
        self.setYRange(-1, 1)
        self.setXRange(-1, 1)
        self.hideAxis('bottom')
        self.hideAxis('left')

    def redraw_state(self, sample):
        pass

class CircleFeedbackProtocolWidget(ProtocolWidget):
    def __init__(self, noise_scaler=100, **kwargs):
        super(CircleFeedbackProtocolWidget, self).__init__(**kwargs)
        self.noise_scaler = noise_scaler
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        self.noise = np.sin(15*self.x)*0.5-0.5
        #self.noise = np.random.uniform(-0.5, 0.5, 100)-0.5
        self.p1 = self.plot(np.sin(self.x),  np.cos(self.x), pen=pg.mkPen(77, 144, 254)).curve
        self.p2 = self.plot(np.sin(self.x), -np.cos(self.x), pen=pg.mkPen(77, 144, 254)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(255, 255, 255, 25))
        self.addItem(fill)

    def redraw_state(self, sample):
        noise_ampl = -np.tanh(sample / self.noise_scaler) + 1
        noise = self.noise*noise_ampl
        self.p1.setData(np.sin(self.x)*(1+noise), np.cos(self.x)*(1+noise))
        self.p2.setData(np.sin(self.x)*(1+noise), -np.cos(self.x)*(1+noise))
        pass

class BaselineProtocolWidget(ProtocolWidget):
    def __init__(self, text='Relax your hands', **kwargs):
        super(BaselineProtocolWidget, self).__init__(**kwargs)
        text_item = pg.TextItem(html='<center><font size="7" color="white">{}</font></center>'.format(text),
                                anchor=(0.5, 0.5))
        text_item.setTextWidth(500)
        self.addItem(text_item)

    def redraw_state(self, sample):
        self.plotItem.setLabel('top', 'mean={:.2f}, std={:.2f}'.format(*sample)) # TODO: delete