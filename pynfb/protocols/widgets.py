import string
import time
import warnings
import random

import cv2
import logging

from PyQt5.QtWidgets import QDesktopWidget, QVBoxLayout

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
    def __init__(self, noise_scaler=2, show_reward=False, radius = 3, circle_border=0, m_threshold=1, r_threshold=0):
        super(BarFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.x = np.linspace(-1, 1, 100)
        self.widget = None
        self.m_threshold = m_threshold
        self.r_threshold = r_threshold

    def prepare_widget(self, widget):
        super(BarFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(229, 223, 213)).curve
        self.p2 = widget.plot(self.x, np.zeros_like(self.x)-5, pen=pg.mkPen(229, 223, 213)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(229, 223, 213, 25))
        self.fill = fill
        self.threshold_line = widget.plot()
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
        self.threshold_line.setData(self.x, np.ones_like(self.x) * self.r_threshold)
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        self.p1.setData(self.x, np.zeros_like(self.x)+max(min(sample, 5), -5))
        self.p2.setData(self.x, np.zeros_like(self.x)-5)

        if sample > self.r_threshold:
            self.fill.setBrush(176, 176, 48, 25)
        else:
            self.fill.setBrush(35, 45, 176, 25)
        pass

class EyeTrackFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, show_reward=False, m_threshold=1, r_threshold=100, center_fixation=0):
        super(EyeTrackFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.fixdot = np.linspace(-np.pi / 2, np.pi / 2, 200)
        self.fixdot_radius = 0.5
        self.widget = None
        self.m_threshold = m_threshold
        self.r_threshold = r_threshold
        self.centre_fixation = center_fixation
        self.eye_range = 580


    def prepare_widget(self, widget):
        self.widget = widget
        self.screen = QDesktopWidget().screenGeometry(widget)
        # Draw fixation dot
        self.p1_fd = self.widget.plot(self.fixdot_radius * np.sin(self.fixdot),
                                      self.fixdot_radius * np.cos(self.fixdot), pen=pg.mkPen(color='black')).curve
        self.p2_fd = self.widget.plot(self.fixdot_radius * np.sin(self.fixdot),
                                      self.fixdot_radius * -np.cos(self.fixdot), pen=pg.mkPen(color='black')).curve
        fill_fd = pg.FillBetweenItem(self.p1_fd, self.p2_fd, brush=('black'))
        self.fill = fill_fd
        widget.addItem(fill_fd)

    def set_red_state(self, flag):
        if flag:
            self.p1_fd.setPen(pg.mkPen(176, 35, 48))
            self.p2_fd.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1_fd.setPen(pg.mkPen(229, 223, 213))
            self.p2_fd.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def redraw_state(self, sample, m_sample):
        # Map the centre fixation to the min and the max fixation to the edge of the screen
        un_scaled_min = self.centre_fixation
        un_scaled_max = self.eye_range/2
        scaled_min = 0
        scaled_max = 5
        un_scaled_range = (un_scaled_max - un_scaled_min)
        scaled_range = (scaled_max - scaled_min)
        scaled_sample = (((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
        # if sample < self.centre_fixation:
        #     un_scaled_min = self.eye_range/2
        #     un_scaled_max = self.centre_fixation
        #     scaled_min = -5
        #     scaled_max = 0
        #     un_scaled_range = (un_scaled_max - un_scaled_min)
        #     scaled_range = (scaled_max - scaled_min)
        #     scaled_sample = (((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min


        # translate the dot in the x direction based on the sample amplitude
        tr = QTransform()  # prepare ImageItem transformation:
        self.scale_factor = 20
        tr.translate(scaled_sample, 0)
        # tr.scale(1./self.scale_factor, 1./self.scale_factor)  # scale horizontal and vertical axes
        self.fill.setTransform(tr)
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

        self.fixdot = np.linspace(-np.pi/2, np.pi/2, 200)
        self.fixdot_radius = 0.1
        self.colour = "black"

    def prepare_widget(self, widget):
        super(GaborFeedbackProtocolWidgetPainter, self).prepare_widget(widget)

        # Draw and align gabor patch
        print("PREPARING GABOR!!!!")
        gabor = GaborPatch(theta=self.gabor_theta)
        self.widget = widget
        self.gabor = gabor
        blurred = cv2.GaussianBlur(gabor, ksize=(0, 0), sigmaX=50)
        self.fill = pg.ImageItem(gabor)
        self.fill.setOpts(update=True, opacity=0)
        tr = QTransform()  # prepare ImageItem transformation:
        self.scale_factor = 20
        self.x_off = gabor.shape[0]/(2*self.scale_factor)
        self.y_off = gabor.shape[1]/(2*self.scale_factor)
        tr.translate(-self.x_off, -self.y_off)
        tr.scale(1./self.scale_factor, 1./self.scale_factor)  # scale horizontal and vertical axes
        self.fill.setTransform(tr)
        self.widget.addItem(self.fill)
        # draw cross
        # self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(color=(0, 0, 0), width=4)).curve
        # self.p2 = widget.plot(np.zeros_like(self.x), self.x, pen=pg.mkPen(color=(0, 0, 0), width=4)).curve

        # Draw fixation dot
        self.p1_fd = self.widget.plot(self.fixdot_radius * np.sin(self.fixdot),
                                      self.fixdot_radius * np.cos(self.fixdot), pen=pg.mkPen(color=self.colour)).curve
        self.p2_fd = self.widget.plot(self.fixdot_radius * np.sin(self.fixdot),
                                      self.fixdot_radius * -np.cos(self.fixdot), pen=pg.mkPen(color=self.colour)).curve
        fill_fd = pg.FillBetweenItem(self.p1_fd, self.p2_fd, brush=(self.colour))
        widget.addItem(fill_fd)

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


class PlotFeedbackWidgetPainter(Painter):
    def __init__(self, m_threshold=1, r_threshold=0, show_reward=False):
        super(PlotFeedbackWidgetPainter, self).__init__(show_reward=show_reward)
        self.widget = None
        self.m_threshold = m_threshold
        self.r_threshold = r_threshold
        self.gabor_theta=0
        self.maxLen = 150  # max number of data points to show on graph
        self.x = np.linspace(-self.maxLen*2, self.maxLen*2, 10)
        self.fixdot = np.linspace(-np.pi / 2, np.pi / 2, 200)
        self.fixdot_radius = 0.02
        self.maxLen = 150  # max number of data points to show on graph

    def prepare_widget(self, widget):
        super(PlotFeedbackWidgetPainter, self).prepare_widget(widget)

        self.widget = widget
        self.dat1 = []
        self.p1 = widget

        self.p1.hideAxis('bottom')
        self.p1.hideAxis('left')
        self.p1.setYRange(-1, 1, padding=0)

        self.curve1 = self.p1.plot()
        self.threshold_line = self.p1.plot()
        r_ratio = self.maxLen/2
        self.p1_fd = self.p1.plot(self.fixdot_radius*r_ratio * np.sin(self.fixdot),
                                      self.fixdot_radius * np.cos(self.fixdot), pen=pg.mkPen(color=(0,0,0)))
        self.p2_fd = self.p1.plot(self.fixdot_radius *r_ratio* np.sin(self.fixdot),
                                      self.fixdot_radius * -np.cos(self.fixdot), pen=pg.mkPen(color=(0,0,0)))


        fill_fd = pg.FillBetweenItem(self.p1_fd, self.p2_fd, brush=(0,0,0))
        self.fill = fill_fd
        self.p1.addItem(fill_fd)


    def set_red_state(self, flag):
        if flag:
            # TODO: figure out what needs to go here
            pass
        else:
            # TODO: figure out what needs to go here
            pass

    def redraw_state(self, sample, m_sample):
        self.threshold_line.setData(self.x, np.ones_like(self.x) * self.r_threshold)
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)

        if len(self.dat1) > self.maxLen:
            self.dat1 = self.dat1[1:]

        pen1 = pg.mkPen(255,233,0)
        self.dat1 = self.dat1 + [sample]
        self.curve1.setData(np.arange(self.maxLen/2 - len(self.dat1), self.maxLen/2), self.dat1, pen=pen1)


class PosnerFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, show_reward=False, m_threshold=1, r_threshold=0, no_nfb=False, max_th=1):
        super(PosnerFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.widget = None
        self.m_threshold = m_threshold
        self.r_threshold = r_threshold
        self.circle = np.linspace(-np.pi / 2, np.pi / 2, 200)
        self.fixdot_radius = 0.06
        self.stim_radius = 0.4
        self.x = np.linspace(-self.stim_radius, self.stim_radius, 100)
        self.train_side = None
        self.stim_side = None
        self.stim = False # when true, remove the cross
        self.test_signal_sample = 0
        self.no_nfb = no_nfb
        self.max_th = max_th
        self.kill = False # flag to allow widget to be killed (after target is displayed)

    def prepare_widget(self, widget):
        super(PosnerFeedbackProtocolWidgetPainter, self).prepare_widget(widget)

        self.widget = widget
        # Draw fixation dot
        self.p1_fd = self.widget.plot(self.fixdot_radius * np.sin(self.circle),
                                      self.fixdot_radius * np.cos(self.circle), pen=pg.mkPen(color=(0,0,0,0))).curve
        self.p2_fd = self.widget.plot(self.fixdot_radius * np.sin(self.circle),
                                      self.fixdot_radius * -np.cos(self.circle), pen=pg.mkPen(color=(0,0,0,0))).curve
        fill_fd = pg.FillBetweenItem(self.p1_fd, self.p2_fd, brush=(0,0,0,0))
        self.fill = fill_fd
        widget.addItem(fill_fd)

        # draw Left and right stim
        left_off_x = -5
        left_off_y = -1
        curve_width = 8
        self.st_l1 = self.widget.plot(left_off_x + self.stim_radius * np.sin(self.circle),
                                     left_off_y + self.stim_radius * np.cos(self.circle), pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve
        self.st_l2 = self.widget.plot(left_off_x + self.stim_radius * np.sin(self.circle),
                                     left_off_y + self.stim_radius * -np.cos(self.circle), pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve
        self.cr_l1 = widget.plot(left_off_x + self.x, left_off_y + np.zeros_like(self.x), pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve
        self.cr_l2 = widget.plot(left_off_x + np.zeros_like(self.x), left_off_y + self.x, pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve

        right_off_x = 5
        right_off_y = -1
        self.st_r1 = self.widget.plot(right_off_x + self.stim_radius * np.sin(self.circle),
                                     right_off_y + self.stim_radius * np.cos(self.circle), pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve
        self.st_r2 = self.widget.plot(right_off_x + self.stim_radius * np.sin(self.circle),
                                     right_off_y + self.stim_radius * -np.cos(self.circle), pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve
        self.cr_r1 = widget.plot(right_off_x + self.x, right_off_y + np.zeros_like(self.x), pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve
        self.cr_r2 = widget.plot(right_off_x + np.zeros_like(self.x), right_off_y + self.x, pen=pg.mkPen(color=(0,0,0,0), width=curve_width)).curve


    def set_red_state(self, flag):
        # TODO: make something alert the participant to eyes wandering
        if flag:
            self.p1.setPen(pg.mkPen(176, 35, 48))
            self.p2.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1.setPen(pg.mkPen(229, 223, 213))
            self.p2.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def over_m_threshold(self, m_sample, flag):
        if flag:
            logging.info(f"EXCESSIVE EYE MOVEMENT! TH: {self.m_threshold}, SAMP: {m_sample}")

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            # self.set_red_state(m_sample > self.m_threshold)
            self.over_m_threshold(m_sample, m_sample > self.m_threshold)
            self.over_m_threshold(m_sample, m_sample < - self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)

        # Ensure the correct side gets trained - this assumes a leftward AAI (L-R)/(L+R)
        if self.train_side == 2:
            sample *= -1

        # Set the fixation dot brush
        self.fill.setBrush('black')
        # Scale the sample to the target colour range
        un_scaled_min = self.r_threshold
        un_scaled_max = self.max_th
        scaled_min = 255
        scaled_max = 0
        un_scaled_range = (un_scaled_max - un_scaled_min)
        scaled_range = (scaled_max - scaled_min)
        scaled_sample =(((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
        scaled_sample = np.clip(scaled_sample, 0, 255)
        scaled_test_sample =(((self.test_signal_sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
        scaled_test_sample = np.clip(scaled_test_sample, 0, 255)

        if self.test_signal_sample < 0:
            scaled_range = -(scaled_max - scaled_min)
            scaled_test_sample = (((self.test_signal_sample - un_scaled_min) * scaled_range) / un_scaled_range) + scaled_min
            scaled_test_sample = np.clip(scaled_test_sample, 0, 255)
            distractor_brush = (255, scaled_test_sample, scaled_test_sample)
        else:
            distractor_brush = (scaled_test_sample, 255, scaled_test_sample)

        if sample < self.r_threshold:
            un_scaled_range = (un_scaled_max - un_scaled_min)
            scaled_range = -(scaled_max - scaled_min)
            scaled_sample = (((sample - un_scaled_min) * scaled_range) / un_scaled_range) + scaled_min
            scaled_sample = np.clip(scaled_sample, 0, 255)

            # Distractor is being focussed on more - colour target red
            target_brush = (255, scaled_sample, scaled_sample)
        else:
            # Target is being focussed on more - colour target green
            target_brush = (scaled_sample, 255, scaled_sample)

        if self.train_side == 1:
            left_brush = target_brush
            right_brush = distractor_brush
        elif self.train_side == 2:
            right_brush = target_brush
            left_brush = distractor_brush
        else:
            right_brush = distractor_brush
            left_brush = distractor_brush

        if self.no_nfb:
            right_brush = distractor_brush
            left_brush = distractor_brush

        curve_width=8
        self.st_l1.setPen(pg.mkPen(color=left_brush, width=curve_width))
        self.st_l2.setPen(pg.mkPen(color=left_brush, width=curve_width))
        if self.stim and self.stim_side == 1:
            logging.debug(f"LEFT STIM: {time.time()*1000}")
            self.cr_l1.setPen(pg.mkPen(color=left_brush, width=curve_width))
            self.cr_l2.setPen(pg.mkPen(color=left_brush, width=curve_width))
        else:
            self.cr_l1.setPen(pg.mkPen(color=(0,0,0,0), width=curve_width))
            self.cr_l2.setPen(pg.mkPen(color=(0,0,0,0), width=curve_width))

        self.st_r1.setPen(pg.mkPen(color=right_brush, width=curve_width))
        self.st_r2.setPen(pg.mkPen(color=right_brush, width=curve_width))
        if self.stim and self.stim_side == 2:
            logging.debug(f"RIGHT STIM: {time.time()*1000}")
            self.cr_r1.setPen(pg.mkPen(color=right_brush, width=curve_width))
            self.cr_r2.setPen(pg.mkPen(color=right_brush, width=curve_width))
        else:
            self.cr_r1.setPen(pg.mkPen(color=(0,0,0,0), width=curve_width))
            self.cr_r2.setPen(pg.mkPen(color=(0,0,0,0), width=curve_width))


class FixationCrossProtocolWidgetPainter(Painter):
    def __init__(self, text="", colour=(0,0,0)):
        super(FixationCrossProtocolWidgetPainter, self).__init__()
        self.x = np.linspace(-0.25, 0.25, 10)
        self.fixdot = np.linspace(-np.pi/2, np.pi/2, 200)
        self.fixdot_radius = 0.1
        self.text = text
        self.text_item = pg.TextItem()
        self.widget = None
        self.colour = colour
        self.text_color = "#e5dfc5"
        self.probe = None
        self.probe_loc = "LEFT"
        self.probe_stim = np.linspace(-np.pi/2, np.pi/2, 200)
        self.probe_radius = 0.1
        self.fixation_type = "dot"

    def prepare_widget(self, widget):
        super(FixationCrossProtocolWidgetPainter, self).prepare_widget(widget)

        self.widget = widget
        self.text_item.setHtml(f'<center><font size="7" color={self.text_color}>{self.text}</font></center>')
        self.text_item.setAnchor((0.5, 0.5))
        self.text_item.setTextWidth(500)
        self.widget.addItem(self.text_item)

        if self.fixation_type == 'cross':
            # draw cross
            self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(color=self.colour, width=4)).curve
            self.p2 = widget.plot(np.zeros_like(self.x), self.x, pen=pg.mkPen(color=self.colour, width=4)).curve
        else:
            # draw fixation dot
            self.p1_fd = self.widget.plot(self.fixdot_radius * np.sin(self.fixdot), self.fixdot_radius * np.cos(self.fixdot), pen=pg.mkPen(color=(0,0,0,0))).curve
            self.p2_fd = self.widget.plot(self.fixdot_radius * np.sin(self.fixdot), self.fixdot_radius * -np.cos(self.fixdot), pen=pg.mkPen(color=(0,0,0,0))).curve
            fill_fd = pg.FillBetweenItem(self.p1_fd, self.p2_fd, brush=(0,0,0,0))
            self.fill_fd = fill_fd
            widget.addItem(fill_fd)

        # Draw probe stim
        self.p1_1 = self.widget.plot(self.probe_radius * np.sin(self.probe_stim), self.probe_radius * np.cos(self.probe_stim), pen=pg.mkPen(0, 0, 0, 0)).curve
        self.p2_1 = self.widget.plot(self.probe_radius * np.sin(self.probe_stim), self.probe_radius * -np.cos(self.probe_stim), pen=pg.mkPen(0, 0, 0, 0)).curve
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

        # colour the fixation dot (Do this here otherwise you get weird flashes between protocols)

        self.fill_fd.setBrush(self.colour)
        # Draw the probe if requested
        if self.probe:
            logging.debug(f"PROBE DRAW START TIME: {time.time()*1000}")
            self.fill.setBrush((0, 0, 0, 255))
            tr = QTransform()
            if self.probe_loc == "LEFT":
                loc = 1
            else:
                loc = -1
            x_off = 5 * loc # should correspond to an eccentricity of 6.7deg
            y_off = -0 # Original paper didn't have y offset
            tr.translate(-x_off, -y_off)
            self.fill.setTransform(tr)
        else:
            self.fill.setBrush((0, 0, 0, 0))


    def set_message(self, text):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))


class PosnerCueProtocolWidgetPainter(Painter):
    def __init__(self, cond=None):
        """
        cond = side to draw cue. 0=left, 1=right, 2=center
        """
        super(PosnerCueProtocolWidgetPainter, self).__init__()
        self.s1 = np.linspace(-0.25, 0, 10)
        self.s2 = np.linspace(0, 0.25, 10)
        self.fixdot = np.linspace(-np.pi/2, np.pi/2, 200)
        self.fixdot_radius = 0.1
        self.cond = cond
        self.widget = None
        self.colour = (0, 0, 0)

    def prepare_widget(self, widget):
        super(PosnerCueProtocolWidgetPainter, self).prepare_widget(widget)

        self.widget = widget

        # Draw the cue outline
        self.p1 = self.widget.plot(self.s1, self.s2, pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p2 = self.widget.plot(self.s2, self.s1, pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p3 = self.widget.plot(self.s1, -self.s2, pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p4 = self.widget.plot(self.s2, -self.s1, pen=pg.mkPen(color=self.colour, width=4)).curve

    def redraw_state(self, sample, m_sample):
        pass

    def left_cue(self):
        # self.p1 = self.widget.plot(self.s1, self.s2, pen=pg.mkPen(color=(0,255,0), width=4)).curve
        # self.p3 = self.widget.plot(self.s1, -self.s2, pen=pg.mkPen(color=(0,255,0), width=4)).curve
        self.p1.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p3.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p2.setPen(pg.mkPen(color=(0,255,0,0), width=4))
        self.p4.setPen(pg.mkPen(color=(0,255,0,0), width=4))

    def right_cue(self):
        # self.p2 = self.widget.plot(self.s2, self.s1, pen=pg.mkPen(color=(0, 255, 0), width=4)).curve
        # self.p4 = self.widget.plot(self.s2, -self.s1, pen=pg.mkPen(color=(0, 255, 0), width=4)).curve
        self.p2.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p4.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p1.setPen(pg.mkPen(color=(0,255,0,0), width=4))
        self.p3.setPen(pg.mkPen(color=(0,255,0,0), width=4))

    def center_cue(self):
        # self.p1 = self.widget.plot(self.s1, self.s2, pen=pg.mkPen(color=(0,255,0), width=4)).curve
        # self.p2 = self.widget.plot(self.s2, self.s1, pen=pg.mkPen(color=(0,255,0), width=4)).curve
        # self.p3 = self.widget.plot(self.s1, -self.s2, pen=pg.mkPen(color=(0,255,0), width=4)).curve
        # self.p4 = self.widget.plot(self.s2, -self.s1, pen=pg.mkPen(color=(0,255,0), width=4)).curve
        self.p2.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p4.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p1.setPen(pg.mkPen(color=(0,255,0), width=4))
        self.p3.setPen(pg.mkPen(color=(0,255,0), width=4))

    def reset_cue(self):
        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        self.p4.clear()
        self.p1 = self.widget.plot(self.s1, self.s2, pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p2 = self.widget.plot(self.s2, self.s1, pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p3 = self.widget.plot(self.s1, -self.s2, pen=pg.mkPen(color=self.colour, width=4)).curve
        self.p4 = self.widget.plot(self.s2, -self.s1, pen=pg.mkPen(color=self.colour, width=4)).curve

    def set_message(self, text):
        self.cond = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.cond))


class BaselineProtocolWidgetPainter(Painter):
    def __init__(self, text='Relax', show_reward=False):
        super(BaselineProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text
        self.text_color = "#e5dfc5"
        self.text_item = pg.TextItem()

    def prepare_widget(self, widget):
        super(BaselineProtocolWidgetPainter, self).prepare_widget(widget)
        self.text_item.setHtml(f'<center><font size="7" color="{self.text_color}">{self.text}</font></center>')
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
    def __init__(self, text='Relax', gabor_theta=45, fs = 0, show_reward=False, previous_score=None):
        super(ParticipantChoiceWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text
        self.text_item = pg.TextItem()
        self.rtext_item = pg.TextItem()
        self.gabor_theta = gabor_theta
        print(f'CHOICE_THETA={gabor_theta}')
        self.previous_score = previous_score
        self.fs = fs
        self.current_sample_idx = 0
        self.show_duration = 1.0


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

        self.text_item.setHtml(f'<center><font size="7" color="#e5dfc5"><p>Is this the orientation you saw? <br>Y (\u2191), N (\u2193)</p></font></center>')
        self.text_item.setAnchor((0.5, -1.8))
        self.text_item.setTextWidth(500)

        self.rtext_item.setHtml(f'<center><font size="7" color="#e5dfc5"><p>Score: {self.previous_score} </p></font></center>')
        self.rtext_item.setAnchor((0.5, 4))
        self.rtext_item.setTextWidth(500)

        self.widget.addItem(self.text_item)
        self.widget.addItem(self.rtext_item)
        self.plotItem = self.widget.plotItem


    def redraw_state(self, sample, m_sample):
        # Display reward
        if self.previous_score:
            self.rtext_item.setHtml(
                f'<center><font size="7" color="#e5dfc5"><p>Score: {self.previous_score} % </p></font></center>')

        # turn the gabor patch off after 0.5 seconds
        if self.current_sample_idx/self.fs > self.show_duration:
            self.fill.setOpts(update=True, opacity=0)

    def set_message(self, text, color="#e5dfc5"):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="{}">{}</font></center>'.format(color, self.text))

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
        self.probe_loc = ["LT", "MT", "RT", 'LM', 'MM', 'RM', 'LB', 'MB', 'RB']
        self.probe_offsets = {"LT": (1, -1), "MT": (0, -1), "RT": (-1, -1),
                              "LM": (1, 0), "MM": (0, 0), "RM": (-1, 0),
                              "LB": (1, 1), "MB": (0, 1), "RB": (-1, 1)}
        random.shuffle(self.probe_loc)
        self.probe_loc.insert(0, "CROSS")
        self.probe_loc.append("CROSS")
        self.prob_loc_indx = 0
        self.position_no = 0
        self.current_sample_idx = 0
        self.probe_radius = 10
        self.position_time = self.protocol_duration/len(self.probe_loc)
        self.probe_stim = np.linspace(-np.pi/2, np.pi/2, 100)
        self.fudge_factor_x = 50 #TODO Figure out why this fudge factor is needed (seems similar for all screens)
        self.fudge_factor_y = 35

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

        # TODO: turn the scale, motion dot, and calibration dot on and off with config (or use separate protocols)
        # draw calibration scale
        # vline = np.linspace(-self.screen.width()/150, self.screen.width()/150, 10)
        # hline = np.linspace(-self.screen.width()/2 - self.fudge_factor_x/2, self.screen.width()/2 + self.fudge_factor_x/2, 10)
        # calibration_scale_hline = widget.plot(hline,np.zeros_like(hline), pen=pg.mkPen(color=(0,0,255), width=4)).curve
        # alphabet = list(string.ascii_uppercase)
        # digits = [0,1,2,3,4,5,6,7,8,9]
        # alphabet_offsets = list(range(round(-self.screen.width()/2 - self.fudge_factor_x/2), round(self.screen.width()/2 + self.fudge_factor_x/2), round((self.screen.width()+self.fudge_factor_x)/(len(alphabet)))))
        # digits_offsets = list(range(round(-self.screen.width()/2 - self.fudge_factor_x/2), round(self.screen.width()/2 + self.fudge_factor_x/2), round((self.screen.width()+self.fudge_factor_x)/(len(digits)-1))))
        # for idx, a in enumerate(alphabet):
        #     calibration_scale_vline = widget.plot(np.zeros_like(vline) + alphabet_offsets[idx], vline-11, pen=pg.mkPen(color=(0,0,255), width=4)).curve
        # for idx, a in enumerate(digits):
        #     calibration_scale_vline = widget.plot(np.zeros_like(vline) + digits_offsets[idx], vline+11, pen=pg.mkPen(color=(0,0,255), width=4)).curve


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
            x_off = (self.screen.width() + self.fudge_factor_x)/2 * offsets[0]
            y_off = (self.screen.height() + self.fudge_factor_y)/2 * offsets[1]
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


