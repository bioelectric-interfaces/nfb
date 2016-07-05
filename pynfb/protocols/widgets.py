import pyqtgraph as pg
import numpy as np
import time


class ProtocolWidget(pg.PlotWidget):
    def __init__(self, **kwargs):
        super(ProtocolWidget, self).__init__(**kwargs)
        width = 5
        self.setYRange(-width, width)
        self.setXRange(-width, width)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setBackgroundBrush(pg.mkBrush('#252120'))
        self.reward_str = '<font size="4" color="#B48375">Reward: </font><font size="5" color="#91C7A9">{}</font>'
        self.reward = pg.TextItem(html=self.reward_str.format(0))
        self.reward.setPos(-4.7, 4.7)
        self.reward.setTextWidth(300)
        self.addItem(self.reward)
        self.clear_all()

    def clear_all(self):
        for item in self.items():
            self.removeItem(item)
        self.addItem(self.reward)

    def update_reward(self, reward):
        self.reward.setHtml(self.reward_str.format(reward))

    def show_reward(self, flag):
        if flag:
            self.reward.show()
        else:
            self.reward.hide()


class CircleFeedbackProtocolWidgetPainter():
    def __init__(self, noise_scaler=2):
        self.noise_scaler = noise_scaler
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        self.noise = np.sin(15*self.x)*0.5-0.5
        #self.noise = np.random.uniform(-0.5, 0.5, 100)-0.5
        self.widget = None

    def prepare_widget(self, widget):
        self.p1 = widget.plot(np.sin(self.x), np.cos(self.x), pen=pg.mkPen(229, 223, 213)).curve
        self.p2 = widget.plot(np.sin(self.x), -np.cos(self.x), pen=pg.mkPen(229, 223, 213)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(229, 223, 213, 25))
        widget.addItem(fill)

    def redraw_state(self, sample):
        if np.ndim(sample)>0:
            sample = np.sum(np.abs(sample))
        noise_ampl = -np.tanh(sample + self.noise_scaler) + 1
        noise = self.noise*noise_ampl
        self.p1.setData(np.sin(self.x)*(1+noise), np.cos(self.x)*(1+noise))
        self.p2.setData(np.sin(self.x)*(1+noise), -np.cos(self.x)*(1+noise))
        pass


class BaselineProtocolWidgetPainter():
    def __init__(self, text='Relax', **kwargs):
        self.text = text

    def prepare_widget(self, widget):
        text_item = pg.TextItem(html='<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text),
                                anchor=(0.5, 0.5))
        text_item.setTextWidth(500)
        widget.addItem(text_item)
        self.plotItem = widget.plotItem

    def redraw_state(self, sample):
        pass


class ThresholdBlinkFeedbackProtocolWidgetPainter():
    def __init__(self, threshold=2000, time_ms=50):
        self.threshold = threshold
        self.time_ms = time_ms
        self.blink_start_time = -1
        self.widget = None
        self.x = np.linspace(-10, 10, 2)
        self.previous_sample = -np.inf

    def prepare_widget(self, widget):
        self.p1 = widget.plot([-10, 10], [10, 10], pen=pg.mkPen(77, 144, 254)).curve
        self.p2 = widget.plot([-10, 10], [-10, -10], pen=pg.mkPen(77, 144, 254)).curve
        self.fill = pg.FillBetweenItem(self.p1, self.p2, brush=(255, 255, 255, 25))
        widget.addItem(self.fill)

    def redraw_state(self, samples):
        samples = np.abs(samples)
        if np.ndim(samples)==0:
            samples = samples.reshape((1,))

        previous_sample = self.previous_sample
        do_blink = False
        for sample in samples:
            if (sample >= self.threshold >= previous_sample) and (self.blink_start_time < 0):
                do_blink = True
            previous_sample = sample

        if do_blink:
            self.blink_start_time = time.time()

        if ((time.time() - self.blink_start_time < self.time_ms * 0.001) and (self.blink_start_time > 0)):
            self.fill.setBrush((255, 255, 255, 255))
        else:
            self.blink_start_time = -1
            self.fill.setBrush((255, 255, 255, 10))


        self.previous_sample = previous_sample
        pass