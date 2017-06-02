import expyriment

from expyriment import control, design, misc
import expyriment.stimuli
import expyriment.stimuli.extras
from PyQt4 import QtGui, QtCore
import numpy as np

from pynfb.helpers.gabor import GaborPatch
from pynfb.helpers.cross import ABCCross


DEBUG = True
BLACK = (0, 0, 0)

def run_exp():
    # expyriment package settings
    control.defaults.initialize_delay = 0

    # init
    gabor1 = GaborPatch(size=500, lambda_=10, theta=45, sigma=50, phase=0.25, position=(500, 0), contrast=0.1)
    cross = ABCCross()
    cross2 = ABCCross(width=1.5)
    cross3 = ABCCross(hide_dot=True)

    exp = design.Experiment(background_colour=BLACK)
    control.initialize(exp)

    # stimuli
    stimulus = expyriment.stimuli.Rectangle((1000, 1000))
    blank = expyriment.stimuli.BlankScreen(colour=BLACK)

    # stimuli preload
    stimulus.preload()
    blank.preload()
    gabor1.preload()
    cross.preload()
    cross2.preload()
    cross3.preload()

    # main loop
    mean = 0
    k = 0
    #for k in range(1, 1000):
    blank.present()
    while True:
        k += 1

        # preload gabor
        present = bool(np.random.randint(0, 2))
        print(present)

        gabor1 = GaborPatch(size=500, lambda_=5, theta=45, sigma=5*10, phase=0.25, position=(-500, 0), contrast=1) #* np.random.uniform(0, 1, 1))
        gabor1.preload()

        # show-hide stimulus
        t = 0
        if present:
            t = gabor1.present(clear=False, update=False)
        t += cross2.present(clear=False, update=True)
        t += cross.present()

        # print time
        if DEBUG:
            mean += t
            print(t, mean / k)

        # wait
        exp.clock.wait(400)

        # detection task
        expyriment.stimuli.TextLine('?', text_size=70, text_colour=(255, 255, 255)).present()
        button, rt = exp.keyboard.wait([misc.constants.K_LEFT, misc.constants.K_RIGHT])

        if DEBUG:
            message = '+' if ((button == misc.constants.K_RIGHT) == present) else '-'
            response = expyriment.stimuli.TextLine(message, text_size=70, text_colour=(255, 255, 255))
            response.present()
            exp.clock.wait(1000)
            response.unload()

        cross.present(clear=True, update=True)

        exp.clock.wait(1000 * np.random.uniform(1.8, 2.4, 1))
        gabor1.unload()


    control.end()


app = QtGui.QApplication([])
w = QtGui.QPushButton('Run exp')
w.show() #showFullScreen()
w.clicked.connect(run_exp)
app.exec_()

