import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QPainter, QPen, QBrush
from PyQt5.QtCore import Qt

class ProtocolWidget(pg.PlotWidget):
    def __init__(self, **kwargs):
        super(ProtocolWidget, self).__init__(**kwargs)
        width = 5
        self.setYRange(-width, width)
        self.setXRange(-width, width)
        size = 500
        self.setMaximumWidth(size)
        self.setMaximumHeight(size)
        self.setMinimumWidth(size)
        self.setMinimumHeight(size)
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


class Painter:
    def __init__(self, show_reward=False):
        self.show_reward = show_reward

    def prepare_widget(self, widget):
        widget.show_reward(self.show_reward)


class CircleFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, noise_scaler=2, show_reward=False, radius = 3, circle_border=0, m_threshold=1):
        super(CircleFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.noise_scaler = noise_scaler
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        np.random.seed(42)
        self.noise = np.sin(15*self.x)*0.5-0.5 if not circle_border else np.random.uniform(-0.5, 0.5, 100)-0.5
        self.widget = None
        self.radius = radius
        self.m_threshold = m_threshold

    def prepare_widget(self, widget):
        super(CircleFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot(np.sin(self.x), np.cos(self.x), pen=pg.mkPen(229, 223, 213)).curve
        self.p2 = widget.plot(np.sin(self.x), -np.cos(self.x), pen=pg.mkPen(229, 223, 213)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(229, 223, 213, 25))
        self.fill = fill
        widget.addItem(fill)

    def set_red_state(self, flag):
        if flag:
            self.p1.setPen(pg.mkPen(176, 35, 48))
            self.p2.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1.setPen(pg.mkPen(229, 223, 213))
            self.p2.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        noise_ampl = -np.tanh(sample + self.noise_scaler) + 1
        noise = self.noise*noise_ampl
        self.p1.setData(self.radius * np.sin(self.x)*(1+noise), self.radius * np.cos(self.x)*(1+noise))
        self.p2.setData(self.radius * np.sin(self.x)*(1+noise), -self.radius * np.cos(self.x)*(1+noise))
        pass


class BarFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, noise_scaler=2, show_reward=False, radius = 3, circle_border=0, m_threshold=1):
        super(BarFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.x = np.linspace(-1, 1, 100)
        self.widget = None
        self.m_threshold = m_threshold

    def prepare_widget(self, widget):
        super(BarFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(229, 223, 213)).curve
        self.p2 = widget.plot(self.x, np.zeros_like(self.x)-5, pen=pg.mkPen(229, 223, 213)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(229, 223, 213, 25))
        self.fill = fill
        widget.addItem(fill)

    def set_red_state(self, flag):
        if flag:
            self.p1.setPen(pg.mkPen(176, 35, 48))
            self.p2.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1.setPen(pg.mkPen(229, 223, 213))
            self.p2.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        self.p1.setData(self.x, np.zeros_like(self.x)+max(min(sample, 5), -5))
        self.p2.setData(self.x, np.zeros_like(self.x)-5)
        pass


class GaborFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, noise_scaler=2, show_reward=False, radius = 3, circle_border=0, m_threshold=1):
        super(GaborFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.noise_scaler = noise_scaler
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        np.random.seed(42)
        self.noise = np.sin(15*self.x)*0.5-0.5 if not circle_border else np.random.uniform(-0.5, 0.5, 100)-0.5
        self.widget = None
        self.radius = radius
        self.m_threshold = m_threshold

    def prepare_widget(self, widget):
        super(GaborFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        gabor = GaborPatch(theta=90)
        self.gabor = gabor
        fill = pg.ImageItem(gabor)
        widget.addItem(fill)

    def set_red_state(self, flag):
        if flag:
            self.p1.setPen(pg.mkPen(176, 35, 48))
            self.p2.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1.setPen(pg.mkPen(229, 223, 213))
            self.p2.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        self.gabor = GaborPatch(theta=sample*180/np.pi)
        pass

class BaselineProtocolWidgetPainter(Painter):
    def __init__(self, text='Relax', show_reward=False):
        super(BaselineProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text
        self.text_item = pg.TextItem()

    def prepare_widget(self, widget):
        super(BaselineProtocolWidgetPainter, self).prepare_widget(widget)
        self.text_item.setHtml(f'<center><font size="7" color="#e5dfc5">{self.text}</font></center>')
        self.text_item.setAnchor((0.5, 0.5))
        self.text_item.setTextWidth(500)
        widget.addItem(self.text_item)
        self.plotItem = widget.plotItem

    def redraw_state(self, sample, m_sample):
        pass

    def set_message(self, text):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))

class ThresholdBlinkFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, threshold=2000, time_ms=50, show_reward=False):
        super(ThresholdBlinkFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.threshold = threshold
        self.time_ms = time_ms
        self.blink_start_time = -1
        self.widget = None
        self.x = np.linspace(-10, 10, 2)
        self.previous_sample = -np.inf

    def prepare_widget(self, widget):
        super(ThresholdBlinkFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot([-10, 10], [10, 10], pen=pg.mkPen(77, 144, 254)).curve
        self.p2 = widget.plot([-10, 10], [-10, -10], pen=pg.mkPen(77, 144, 254)).curve
        self.fill = pg.FillBetweenItem(self.p1, self.p2, brush=(255, 255, 255, 25))
        widget.addItem(self.fill)

    def redraw_state(self, samples, m_sample):
        samples = np.abs(samples)
        if np.ndim(samples)==0:
            samples = samples.reshape((1, ))

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


class VideoProtocolWidgetPainter(Painter):
    def __init__(self, video_file_path):
        super(VideoProtocolWidgetPainter, self).__init__()
        self.widget = None
        self.video = None
        self.timer = time.time()
        self.timer_period = 1 / 30
        self.frame_counter = 0
        self.n_frames = None
        self.err_msg = "Could't open video file. "
        import os.path
        if os.path.isfile(video_file_path):
            try:
                import imageio as imageio
                self.video = imageio.get_reader(video_file_path,  'ffmpeg')
                self.n_frames = self.video.get_length() - 1
            except ImportError as e:
                print(e.msg)
                self.err_msg += e.msg
        else:
            self.err_msg = "No file {}".format(video_file_path)


    def prepare_widget(self, widget):
        super(VideoProtocolWidgetPainter, self).prepare_widget(widget)
        if self.video is not None:
            self.img = pg.ImageItem()
            self.img.setScale(10 / self.video.get_data(0).shape[1])
            self.img.rotate(-90)
            self.img.setX(-5)
            self.img.setY(5/self.video.get_data(0).shape[1]*self.video.get_data(0).shape[0])
            widget.addItem(self.img)

        else:
            text_item = pg.TextItem(html='<center><font size="6" color="#a92f41">{}'
                                         '</font></center>'.format(self.err_msg),
                                    anchor=(0.5, 0.5))
            text_item.setTextWidth(500)
            widget.addItem(text_item)

    def redraw_state(self, sample, m_sample):
        if self.video is not None:
            timer = time.time()
            if timer - self.timer > self.timer_period:
                self.timer = timer
                self.frame_counter = (self.frame_counter + 1) % self.n_frames
                self.img.setImage(self.video.get_data(self.frame_counter))
            pass

def GaborPatch(position=None,
                 sigma=55,
                 theta=35,
                 lambda_=12.5,
                 phase=0.5,
                 psi=120,
                 gamma=1,
                 background_colour=(127, 127, 127)):
    """A class implementing a Gabor Patch.
    From expyriment: https://github.com/expyriment/expyriment-stash/blob/master/extras/expyriment_stimuli_extras/gaborpatch/_gaborpatch.py
    """


    """Create a Gabor Patch.
    Parameters
    ----------
    position  : (int, int), optional
        position of the mask stimulus
    sigma : int or float, optional
        gaussian standard deviation (in pixels) (default=20)
    theta : int or float, optional
        Grating orientation in degrees (default=35)
    lambda_ : int, optional
        Spatial frequency (pixel per cycle) (default=10)
    phase : float
        0 to 1 inclusive (default=.5)
    psi : int, optional
        0 to 1 inclusive (default=1)
    gamma : float
        0 to 1 inclusive (default=1)
    background_colour : (int,int,int), optional
        colour of the background, default: (127, 127, 127)
    Notes
    -----
    The background colour of the stimulus depends of the parameters of
    the Gabor patch and can be determined (e.g. for plotting) with the
    property `GaborPatch.background_colour`.
    """


    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3
    theta = theta / 180.0 * np.pi
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    pattern = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * np.pi / lambda_ * x_theta + psi)

    # make numpy pixel array
    bkg = np.ones((pattern.shape[0], pattern.shape[1], 3)) * \
                                    (np.ones((pattern.shape[1], 3)) * background_colour) #background
    modulation = np.ones((3, pattern.shape[1], pattern.shape[0])) * \
                                    ((255/2.0) * phase * np.ones(pattern.shape) * pattern)  # alpha

    # self._pixel_array = bkg + modulation.T
    # self._pixel_array[self._pixel_array<0] = 0
    # self._pixel_array[self._pixel_array>255] = 255

    # make stimulus
    # Canvas.__init__(self, size=pattern.shape, position=position, colour=background_colour)
    # self._background_colour = background_colour
    return pattern


if __name__ == '__main__':
    from PyQt5 import QtGui, QtWidgets
    from PyQt5 import QtCore, QtWidgets
    a = QtWidgets.QApplication([])
    w = ProtocolWidget()
    w.show()
    b = BarFeedbackProtocolWidgetPainter()
    b.prepare_widget(w)
    timer = QtCore.QTimer()
    timer.start(1000/30)
    timer.timeout.connect(lambda: b.redraw_state(np.random.normal(scale=3), np.random.normal(scale=0.1)))
    a.exec_()
    #for k in range(10000):
    #    sleep(1/30)
    #    b.redraw_state(np.random.normal(size=1))


