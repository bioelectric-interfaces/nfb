import time
import warnings
import random

import cv2
import skimage.io
import skimage.filters

from PyQt5.QtWidgets import QDesktopWidget

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QPainter, QPen, QBrush, QTransform
from PyQt5.QtCore import Qt

class ProtocolWidget(pg.PlotWidget):
    def __init__(self, fbtype=None, size=500, **kwargs):
        super(ProtocolWidget, self).__init__(**kwargs)
        width = 5
        self.setYRange(-width, width)
        self.setXRange(-width, width)
        self.setMaximumWidth(size)
        self.setMaximumHeight(size)
        self.setMinimumWidth(size)
        self.setMinimumHeight(size)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setBackgroundBrush(pg.mkBrush('#252120'))
        # if type and type == "Gabor":
        self.setBackgroundBrush(pg.mkBrush(126,126,126))
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
    def __init__(self, gabor_theta=45, m_threshold=1, r_threshold=0, show_reward=False):
        super(GaborFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        np.random.seed(42)
        self.x = np.linspace(-0.25, 0.25, 10)
        self.widget = None
        self.gabor_theta = gabor_theta
        self.m_threshold = m_threshold
        self.r_threshold = r_threshold
        print(f'GABOR_THETA={gabor_theta}')

    def prepare_widget(self, widget):
        super(GaborFeedbackProtocolWidgetPainter, self).prepare_widget(widget)

        # Draw and align gabor patch
        gabor = GaborPatch(theta=self.gabor_theta)
        self.widget = widget
        self.gabor = gabor
        blurred = cv2.GaussianBlur(gabor, ksize=(0, 0), sigmaX=50)
        self.fill = pg.ImageItem(gabor)
        tr = QTransform()  # prepare ImageItem transformation:
        self.scale_factor = 20
        self.x_off = gabor.shape[0]/(2*self.scale_factor)
        self.y_off = gabor.shape[1]/(2*self.scale_factor)
        tr.translate(-self.x_off, -self.y_off)
        tr.scale(1./self.scale_factor, 1./self.scale_factor)  # scale horizontal and vertical axes
        self.fill.setTransform(tr)
        self.widget.addItem(self.fill)
        # draw cross
        self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(color=(0, 0, 0), width=4)).curve
        self.p2 = widget.plot(np.zeros_like(self.x), self.x, pen=pg.mkPen(color=(0, 0, 0), width=4)).curve

    def set_red_state(self, flag):
        if flag:
            # TODO: figure out what needs to go here
            pass
        else:
            # TODO: figure out what needs to go here
            pass

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        # print(f"SAMPLE: {sample}, ANGLE: {sample*180/np.pi}")
        # scale the values between the r_threshold and 1 to be between 0 and 1
        un_scaled_min = self.r_threshold
        un_scaled_max = 1
        scaled_min = 0
        scaled_max = 1
        un_scaled_range = (un_scaled_max - un_scaled_min)
        scaled_range = (scaled_max - scaled_min)
        scaled_sample = (((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min

        # Change visibility based on opacity
        self.fill.setOpts(update=True, opacity=max(min(scaled_sample, 1.0), 0))

        # change visibilty based on blur
        # sigma (std) of gausiann should be between 100 and 0 where 0 = full visibility
        un_scaled_min = self.r_threshold
        un_scaled_max = 1
        scaled_min = 100
        scaled_max = 0
        un_scaled_range = (un_scaled_max - un_scaled_min)
        scaled_range = (scaled_max - scaled_min)
        scaled_sample =(((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
        # scaled_sample = max(min(scaled_sample, 1.0), 0)

        # self.x_off = self.gabor.shape[0]/(2*self.scale_factor)
        # self.y_off = self.gabor.shape[1]/(2*self.scale_factor)
        # tr = QTransform()
        # tr.translate(self.x_off, self.y_off)
        # tr.scale(self.scale_factor, self.scale_factor)  # scale horizontal and vertical axes
        # self.fill.setTransform(tr)

        # blurred = cv2.blur(self.gabor, ksize=(int(scaled_sample), int(scaled_sample)))
        # self.fill.setImage(blurred)

        ## Change visibility based on gabor filter
        # un_scaled_min = self.r_threshold
        # un_scaled_max = 1
        # scaled_min = self.gabor_theta-90
        # scaled_max = self.gabor_theta
        # un_scaled_range = (un_scaled_max - un_scaled_min)
        # scaled_range = (scaled_max - scaled_min)
        # scaled_sample =(((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
        # gabor = GaborPatch(theta=scaled_sample)
        # blurred = cv2.filter2D(self.gabor, cv2.CV_8UC3, gabor)


class FixationCrossProtocolWidgetPainter(Painter):
    def __init__(self, text="", colour=(0,0,0)):
        super(FixationCrossProtocolWidgetPainter, self).__init__()
        self.x = np.linspace(-0.25, 0.25, 10)
        self.text = text
        self.text_item = pg.TextItem()
        self.widget = None
        self.colour = colour
        self.probe = None
        self.probe_loc = "LEFT"
        self.probe_stim = np.linspace(-np.pi/2, np.pi/2, 100)# TODO: set the size to correspond to lab monitor - should be 0.25 degress

    def prepare_widget(self, widget):
        super(FixationCrossProtocolWidgetPainter, self).prepare_widget(widget)

        self.widget = widget
        self.text_item.setHtml(f'<center><font size="7" color="#e5dfc5">{self.text}</font></center>')
        self.text_item.setAnchor((0.5, 0.5))
        self.text_item.setTextWidth(500)
        self.widget.addItem(self.text_item)

        # draw cross
        self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p2 = widget.plot(np.zeros_like(self.x), self.x, pen=pg.mkPen(color=self.colour, width=4)).curve

        # Draw probe stim
        self.p1_1 = self.widget.plot(np.sin(self.probe_stim), np.cos(self.probe_stim), pen=pg.mkPen(0, 0, 0, 0)).curve
        self.p2_1 = self.widget.plot(np.sin(self.probe_stim), -np.cos(self.probe_stim), pen=pg.mkPen(0, 0, 0, 0)).curve
        fill = pg.FillBetweenItem(self.p1_1, self.p2_1, brush=(0, 0, 0, 0))
        self.fill = fill
        widget.addItem(fill)

    def set_red_state(self, flag):
        if flag:
            # TODO: figure out what needs to go here
            pass
        else:
            # TODO: figure out what needs to go here
            pass

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)

        # Draw the probe if requested
        if self.probe:
            self.fill.setBrush((0, 0, 0, 100))
            tr = QTransform()
            if self.probe_loc == "LEFT":
                loc = 1
            else:
                loc = -1
            x_off = 25 * loc # TODO: scale these based on lab monitor properties - should correspond to an eccentricity of 6.7deg
            y_off = -25 # TODO: check that this should be elevated or at 0
            tr.scale(0.1, 0.1)
            tr.translate(-x_off, -y_off)
            self.fill.setTransform(tr)
        else:
            self.fill.setBrush((0, 0, 0, 0))


    def set_message(self, text):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))

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


class ParticipantInputWidgetPainter(Painter):
    """
    Protocol that waits for participant to press space to continue
    """
    def __init__(self, text='Relax', show_reward=False):
        super(ParticipantInputWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text
        self.text_item = pg.TextItem()

    def prepare_widget(self, widget):
        super(ParticipantInputWidgetPainter, self).prepare_widget(widget)
        score = widget.reward.toPlainText().split(":")[1]
        self.text_item.setHtml(f'<center><font size="7" color="#e5dfc5">score:{score}<br>SPACE TO CONTINUE</font></center>')
        self.text_item.setAnchor((0.5, 0.5))
        self.text_item.setTextWidth(500)
        widget.addItem(self.text_item)
        self.plotItem = widget.plotItem

    def redraw_state(self, sample, m_sample):
        pass

    def set_message(self, text):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))

class ParticipantChoiceWidgetPainter(Painter):
    """
    Protocol for 2-alternative forced choice task (currently for gabor patch specifically)
    TODO: make this generic, i.e. not only for gabor patch
    """
    def __init__(self, text='Relax', gabor_theta=45, show_reward=False, previous_score=None):
        super(ParticipantChoiceWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text
        self.text_item = pg.TextItem()
        self.rtext_item = pg.TextItem()
        self.gabor_theta = gabor_theta
        print(f'CHOICE_THETA={gabor_theta}')
        self.previous_score = previous_score


    def prepare_widget(self, widget):
        super(ParticipantChoiceWidgetPainter, self).prepare_widget(widget)

        self.widget = widget
        score = widget.reward.toPlainText().split(":")[1]

        gabor = GaborPatch(theta=self.gabor_theta)
        self.gabor = gabor
        self.fill = pg.ImageItem(gabor)
        tr = QTransform()  # prepare ImageItem transformation:
        scale_factor = 20
        x_off = gabor.shape[0]/(2*scale_factor)
        y_off = gabor.shape[1]/(2*scale_factor)
        tr.translate(-x_off, -y_off)
        tr.scale(1./scale_factor, 1./scale_factor)  # scale horizontal and vertical axes
        self.fill.setTransform(tr)
        self.widget.addItem(self.fill)

        self.text_item.setHtml(f'<center><font size="7" color="#e5dfc5"><p>Is this the image you saw? <br>Y (\u2191), N (\u2193)</p></font></center>')
        self.text_item.setAnchor((0.5, -1.75))
        self.text_item.setTextWidth(500)

        self.rtext_item.setHtml(f'<center><font size="7" color="#e5dfc5"><p>Score: {self.previous_score} </p></font></center>')
        self.rtext_item.setAnchor((0.5, 5))
        self.rtext_item.setTextWidth(500)

        self.widget.addItem(self.text_item)
        self.widget.addItem(self.rtext_item)
        self.plotItem = self.widget.plotItem


    def redraw_state(self, sample, m_sample):
        # Display reward
        if self.previous_score:
            self.rtext_item.setHtml(
                f'<center><font size="7" color="#e5dfc5"><p>Score: {self.previous_score} </p></font></center>')

    def set_message(self, text):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))

class ExperimentStartWidgetPainter(Painter):
    def __init__(self, text='Relax', show_reward=False):
        super(ExperimentStartWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text
        self.text_item = pg.TextItem()

    def prepare_widget(self, widget):
        super(ExperimentStartWidgetPainter, self).prepare_widget(widget)
        score = widget.reward.toPlainText().split(":")[1]
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


class EyeCalibrationProtocolWidgetPainter(Painter):
    """
    Calibration for eye movements for both eye tracking tasks and eye movement rejection
    """
    def __init__(self, protocol_duration=0):
        super(EyeCalibrationProtocolWidgetPainter, self).__init__()
        self.x = np.linspace(-0.25, 0.25, 10)
        self.protocol_duration = protocol_duration
        self.widget = None
        self.probe_loc = ["LEFT", "RIGHT", "TOP", "BOTTOM"]
        self.probe_offsets = {"LEFT": (1, 0), "RIGHT": (-1, 0), "TOP": (0, -1), "BOTTOM": (0, 1)}
        random.shuffle(self.probe_loc)
        self.probe_loc.insert(0, "CROSS")
        self.probe_loc.append("CROSS")
        self.prob_loc_indx = 0
        self.position_no = 0
        self.current_sample_idx = 0
        self.probe_radius = 10
        self.position_time = self.protocol_duration/len(self.probe_loc)
        self.probe_stim = np.linspace(-np.pi/2, np.pi/2, 100)# TODO: set the size to correspond to lab monitor - should be 0.25 degress

    def prepare_widget(self, widget):
        super(EyeCalibrationProtocolWidgetPainter, self).prepare_widget(widget)

        self.widget = widget

        self.screen = QDesktopWidget().screenGeometry(widget)
        self.x = np.linspace(-self.screen.width()/75, self.screen.width()/75, 10)
        self.probe_stim = np.linspace(-np.pi/2, np.pi/2, 100)

        # draw cross
        self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(color=(0,0,0), width=4)).curve
        self.p2 = widget.plot(np.zeros_like(self.x), self.x, pen=pg.mkPen(color=(0,0,0), width=4)).curve

        # Draw probe stim
        self.p1_1 = self.widget.plot(self.probe_radius * np.sin(self.probe_stim), self.probe_radius * np.cos(self.probe_stim), pen=pg.mkPen(0, 0, 0, 0)).curve
        self.p2_1 = self.widget.plot(self.probe_radius * np.sin(self.probe_stim), self.probe_radius * -np.cos(self.probe_stim), pen=pg.mkPen(0, 0, 0, 0)).curve
        fill = pg.FillBetweenItem(self.p1_1, self.p2_1, brush=(0, 0, 0, 0))
        self.fill = fill
        widget.addItem(fill)

    def set_red_state(self, flag):
        pass

    def redraw_state(self, sample, m_sample):
        if np.ndim(sample)>0:
            sample = np.sum(sample)

        if self.probe_loc[self.position_no] != "CROSS":
            # Hide the cross
            self.p1.setPen(color=(0,0,0, 0), width=4)
            self.p2.setPen(color=(0,0,0, 0), width=4)
            tr = QTransform()
            self.fill.setBrush((0, 0, 0, 255))
            offsets = self.probe_offsets[self.probe_loc[self.position_no]]
            fudge_factor = 50 #TODO CHANGE THIS FUDGE FACTOR FOR THE LAB SCREEN
            x_off = (self.screen.width() + fudge_factor)/2 * offsets[0]
            y_off = (self.screen.height() + fudge_factor)/2 * offsets[1]
            tr.translate(-x_off, -y_off)
            self.fill.setTransform(tr)
        else:
            # Show the cross
            self.p1.setPen(color=(0,0,0, 255), width=4)
            self.p2.setPen(color=(0,0,0, 255), width=4)
            # Hide probe
            self.fill.setBrush((0, 0, 0, 0))

        # Cycle through probe location. start with only cross at start and end (6 positions total)
        if self.current_sample_idx > self.position_time * (1+self.position_no) and self.position_no < len(self.probe_loc)-1:
            self.position_no = self.position_no + 1


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


class ImageProtocolWidgetPainter(Painter):
    def __init__(self, image_file_path):
        super(ImageProtocolWidgetPainter, self).__init__()
        self.widget = None
        self.image = None
        self.err_msg = f"Couldn't open image file at {image_file_path} "
        import os.path
        if os.path.isfile(image_file_path):
            try:
                import imageio as imageio
                self.image = imageio.imread(image_file_path)
            except ImportError as e:
                print(e.msg)
                self.err_msg += e.msg
        else:
            self.err_msg = "No file {}".format(image_file_path)


    def prepare_widget(self, widget):
        super(ImageProtocolWidgetPainter, self).prepare_widget(widget)
        if self.image is not None:
            self.img = pg.ImageItem()
            tr = QTransform()  # prepare ImageItem transformation:
            size = widget.geometry()
            screen = QDesktopWidget().screenGeometry(widget)
            margins = widget.contentsMargins()
            print(f"MARGINS3 left: {margins.left()}, right: {margins.right()}, top: {margins.top()}, bottom: {margins.bottom()}")
            print(f"SCREEN2 H: {screen.height()}, SCREEN2 W: {screen.width()}")
            print(f"IMAGE H: {self.image.shape[0]}, W: {self.image.shape[1]}")
            scale_factor_h = self.image.shape[0]/(screen.height()+40+40) # THIS FUDGE FACTOR AT THE END GETS THE IMAGE MORE OR LESS TO THE EDGES - HAS TO BE CHECKED - WHERE DO THESE VALUES COME FROM?
            scale_factor_w = self.image.shape[1]/(screen.width()+40+40)
            print(f"H: {size.height()}, scalefh: {scale_factor_h}, scalefw: {scale_factor_w}")
            tr.rotate(-90)
            tr.scale(1. / scale_factor_h, 1. / scale_factor_w)  # scale horizontal and vertical axes
            x_off = self.image.shape[0] / (2)
            y_off = self.image.shape[1] / (2)
            tr.translate(-x_off, -y_off)
            self.img.setTransform(tr)
            self.img.setImage(self.image)
            widget.addItem(self.img)

        else:
            text_item = pg.TextItem(html='<center><font size="6" color="#a92f41">{}'
                                         '</font></center>'.format(self.err_msg),
                                    anchor=(0.5, 0.5))
            text_item.setTextWidth(500)
            widget.addItem(text_item)

    def redraw_state(self, sample, m_sample):
        if self.image is not None:
            pass

def GaborPatch(position=None,
                 sigma=25,
                 theta=35,
                 lambda_=12.5,
                 phase=0.5,
                 psi=120,
                 gamma=1,
                 background_colour=(0, 0, 0)):
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
    nstds = 5
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

    pixel_array = bkg + modulation.T
    # self._pixel_array[self._pixel_array<0] = 0
    # self._pixel_array[self._pixel_array>255] = 255

    # make stimulus
    # Canvas.__init__(self, size=pattern.shape, position=position, colour=background_colour)
    # self._background_colour = background_colour


    return pixel_array

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


