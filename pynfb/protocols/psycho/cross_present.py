import expyriment
from time import sleep
from expyriment import control, design, misc
import expyriment.stimuli
import expyriment.stimuli.extras
from PyQt4 import QtGui, QtCore
import numpy as np

from pynfb.helpers.gabor import GaborPatch
from pynfb.helpers.cross import ABCCross


DEBUG = True
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class PsyExperiment2:
    def __init__(self, detection_task=False, feedback=True):
        # setup experiment
        control.defaults.initialize_delay = 0
        control.defaults.window_mode = True
        self.exp = design.Experiment(background_colour=BLACK)

        # init stimulus
        self.gabor = self.set_gabor(-1, 0.5)
        self.cross = ABCCross()
        self.cross2 = ABCCross(width=1.5)

        # trial counter
        self.trial_counter = 0

        # flags
        self.detection = detection_task
        self.feedback = feedback
        pass

    def set_gabor(self, position, contrast, size=500, lambda_=10, theta=45, sigma=50, phase=0.25):
        position = (-500, 0) if position < 0 else (500, 0)
        self.gabor = GaborPatch(size=500, lambda_=10, theta=45, sigma=50, phase=0.25,
                                position=position, contrast=contrast)
        return self.gabor

    def run(self):
        control.initialize(self.exp)
        self.gabor.preload()
        self.cross.preload()
        self.cross2.preload()

        # prepare stimulus
        prepare = expyriment.stimuli.TextLine('Prepare', text_size=50, text_colour=WHITE)
        prepare.present()
        prepare.present()
        pass

    def trial(self):
        if self.trial_counter == 0:
            self.cross.present()

        self.trial_counter += 1

        # cross waiting
        self.cross.present(clear=True, update=True)
        self.exp.clock.wait(1000 * np.random.uniform(1.8, 2.4, 1))

        # present + hide gabor and change cross
        present = bool(np.random.randint(0, 2))
        t = 0
        if present:
            t = self.gabor.present(clear=False, update=False)
        t += self.cross2.present(clear=False, update=True)
        t += self.cross.present()

        # print time
        if DEBUG:
            print(t)
        self.exp.clock.wait(400)

        # detection task
        if self.detection:
            expyriment.stimuli.TextLine('?', text_size=70, text_colour=(255, 255, 255)).present()
            button, rt = self.exp.keyboard.wait([misc.constants.K_LEFT, misc.constants.K_RIGHT])
            if self.feedback:
                message = '+' if ((button == misc.constants.K_RIGHT) == present) else '-'
                response = expyriment.stimuli.TextLine(message, text_size=70, text_colour=(255, 255, 255))
                response.present()
                self.exp.clock.wait(1000)
                response.unload()

    def close(self):
        control.end()



class PsyExperiment:
    def __init__(self, exp, detection_task=False, feedback=True):
        self.exp = exp

        # init stimulus
        self.gabor = self.set_gabor(-1, 0.5)
        self.cross = ABCCross()
        self.cross2 = ABCCross(width=1.5)

        # trial counter
        self.trial_counter = 0

        # flags
        self.detection = detection_task
        self.feedback = feedback
        pass

    def set_gabor(self, position, contrast, size=500, lambda_=10, theta=45, sigma=50, phase=0.25):
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

    def trial(self):
        if self.trial_counter == 0:
            self.cross.present()

        self.trial_counter += 1

        # cross waiting
        self.cross.present(clear=True, update=True)
        self.exp.clock.wait(1000 * np.random.uniform(1.8, 2.4, 1))

        # present + hide gabor and change cross
        present = bool(np.random.randint(0, 2))
        t = 0
        if present:
            t = self.gabor.present(clear=False, update=False)
        t += self.cross2.present(clear=False, update=True)
        t += self.cross.present()

        # print time
        if DEBUG:
            print(t)
        self.exp.clock.wait(400)

        # detection task
        if self.detection:
            expyriment.stimuli.TextLine('?', text_size=70, text_colour=(255, 255, 255)).present()
            button, rt = self.exp.keyboard.wait([misc.constants.K_LEFT, misc.constants.K_RIGHT])
            if self.feedback:
                message = '+' if ((button == misc.constants.K_RIGHT) == present) else '-'
                response = expyriment.stimuli.TextLine(message, text_size=70, text_colour=(255, 255, 255))
                response.present()
                self.exp.clock.wait(1000)
                response.unload()

    def close(self):
        control.end()



if __name__ == '__main__':
    def run_exp():
        exper = PsyExperiment2()
        exper.run()
        sleep(5)
        while True:
           exper.trial()
        control.end()


    app = QtGui.QApplication([])
    w = QtGui.QPushButton('Run exp')
    w.show() #showFullScreen()
    w.clicked.connect(run_exp)
    app.exec_()

