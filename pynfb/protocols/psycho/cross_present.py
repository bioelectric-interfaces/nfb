import expyriment
from time import sleep, time
from expyriment import control, design, misc
import expyriment.stimuli
import expyriment.stimuli.extras
from PyQt4 import QtGui, QtCore
import numpy as np
from pynfb.helpers.gabor import GaborPatch
from pynfb.helpers.cross import ABCCross
import pygame


DEBUG = True
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)



class PsyExperiment:
    def __init__(self, exp, detection_task=False, feedback=True):
        self.exp = exp

        # init stimulus
        self.gabor = self.set_gabor(-1, 0.5)
        self.cross = ABCCross()
        self.cross2 = ABCCross(width=1.5)
        self.present = False

        # trial counter
        self.trial_counter = 0

        # flags
        self.detection = detection_task
        self.feedback = feedback

        # timing
        self.timing = {'prestim_min': 1800, 'prestim_max': 2400, 'poststim': 400, 'response': 400}
        self.t_full = sum(self.timing.values()) - self.timing['prestim_min'] + 32
        self.t_wait_start = None
        self.t_wait = None
        self.is_waiting = False
        self.presentation_sequence = [self.present_pre_stimulus, self.wait_prestim, self.present_stimulus,
                                      self.wait_random]
        self.detection_task_sequence = [self.present_stimulus, self.wait_random, self.run_detection_task,
                                        self.present_pre_stimulus, self.wait_inf]
        self.sequence = self.detection_task_sequence if detection_task else self.presentation_sequence
        self.present_stimulus_index = self.sequence.index(self.present_stimulus)
        self.current_action = 0
        self.current_sample = 0
        pass

    def wait_prestim(self):
        t_wait = self.timing['poststim']
        if not self.is_waiting:
            self.is_waiting = True
            self.t_wait_start = time()*1000
        else:
            if time()*1000 - self.t_wait_start > t_wait:
                self.is_waiting = False

    def wait_inf(self):
        if not self.is_waiting:
            self.is_waiting = True
        else:
            if self.current_sample > 40000:
                self.is_waiting = False

    def wait_random(self):
        if not self.is_waiting:
            self.t_wait = np.random.randint(self.timing['prestim_min'], self.timing['prestim_max'])
            self.is_waiting = True
            self.t_wait_start = time() * 1000
        else:
            if time() * 1000 - self.t_wait_start > self.t_wait:
                self.is_waiting = False

    def run_trial(self, sample):
        self.current_sample = sample
        print(self.current_sample)
        self.sequence[self.current_action]()
        stimulus_presented = self.current_action == self.present_stimulus_index
        if not self.is_waiting:
            self.current_action = (self.current_action + 1) % len(self.sequence)
        return stimulus_presented



    def set_gabor(self, position, contrast):
        position = (-500, 0) if position < 0 else (500, 0)
        self.gabor = GaborPatch(size=500, lambda_=10, theta=45, sigma=50, phase=0.25,
                                position=position, contrast=contrast)
        return self.gabor

    def run(self):
        self.gabor.preload()
        self.cross.preload()
        self.cross2.preload()

        # prepare stimulus
        from time import time
        prepare = expyriment.stimuli.TextLine('Prepare'+str(time()), text_size=50, text_colour=WHITE)
        prepare.present()
        prepare.present()
        pass

    def preload_stimuli(self):
        self.gabor.preload()
        self.cross.preload()
        self.cross2.preload()

    def present_pre_stimulus(self):
        self.cross.present(clear=True, update=True)

    def present_stimulus(self):
        # present + hide gabor and change cross
        self.present = bool(np.random.randint(0, 2))
        t = 0
        if self.present:
            t = self.gabor.present(clear=False, update=False)
        t += self.cross2.present(clear=False, update=True)
        t += self.cross.present()

        # print time
        if DEBUG:
            print(t)

    def run_detection_task1(self):
        # detection task
        print(self.detection)
        if self.detection:
            expyriment.stimuli.TextLine('?', text_size=70, text_colour=(255, 255, 255)).present()
            button, rt = self.exp.keyboard.wait([misc.constants.K_LEFT, misc.constants.K_RIGHT])
            if self.feedback:
                message = '+' if ((button == misc.constants.K_RIGHT) == self.present) else '-'
                response = expyriment.stimuli.TextLine(message, text_size=70, text_colour=(255, 255, 255))
                response.present()
                self.exp.clock.wait(self.timing['response'])
                response.unload()

    def trial(self):
        if self.exp.is_initialized:
            t_trial_start = time()
            if self.trial_counter == 0:
                self.run()
                self.cross.present()

            self.trial_counter += 1

            # cross waiting
            self.present_pre_stimulus()
            self.exp.clock.wait(np.random.randint(self.timing['prestim_min'], self.timing['prestim_max']))

            self.present_stimulus()
            self.exp.clock.wait(self.timing['poststim'])
            # detection task
            self.run_detection_task()
            return time() - t_trial_start

    def run_detection_task(self):
        if self.detection:
            if not self.is_waiting:
                expyriment.stimuli.TextLine('?', text_size=70, text_colour=(255, 255, 255)).present()
                self.is_waiting = True
                self.t_wait_start = time()*1000
                pygame.event.get()
            else:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and  event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        if event.key == pygame.K_LEFT:
                            print("Hey, you pressed the key, 'left'!")
                        if event.key == pygame.K_RIGHT:
                            print("Hey, you pressed the key, 'right'!")
                        if self.feedback:
                            message = '+' if ((event.key == pygame.K_RIGHT) == self.present) else '-'
                            response = expyriment.stimuli.TextLine(message, text_size=70, text_colour=(255, 255, 255))
                            response.present()
                            self.exp.clock.wait(self.timing['response'])
                            response.unload()
                            self.is_waiting = False
                if time()*1000 - self.t_wait_start > 5000:
                    self.is_waiting = False

    def close(self):
        control.end()



if __name__ == '__main__':
    def run_exp():
        control.defaults.initialize_delay = 0
        #control.defaults.window_mode = True
        exp_env = design.Experiment(background_colour=BLACK)
        control.initialize(exp_env)
        exp = PsyExperiment(exp_env, detection_task=True)
        exp.preload_stimuli()
        #sleep(5)

        while True:
            exp.run_trial()

        control.end()


    app = QtGui.QApplication([])
    w = QtGui.QPushButton('Run exp')
    w.show() #showFullScreen()
    w.clicked.connect(run_exp)
    app.exec_()

