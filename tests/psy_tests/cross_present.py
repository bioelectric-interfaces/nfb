import expyriment

from expyriment import control, design, misc
import expyriment.stimuli
import expyriment.stimuli.extras
from PyQt4 import QtGui, QtCore
import numpy as np

from pynfb.helpers.gabor import GaborPatch
from pynfb.helpers.cross import ABCCross


def run_exp():
    #control.run_test_suite()
    #control.set_develop_mode(True)

    # init
    control.defaults.initialize_delay = 0
    gabor1 = GaborPatch(size=500, lambda_=10, theta=45, sigma=50, phase=0.25, position=(500, 0), contrast=0.1)
    cross = ABCCross()
    cross2 = ABCCross(width=1.5)
    cross3 = ABCCross(hide_dot=True)

    exp = design.Experiment(background_colour=gabor1._background_colour)
    control.initialize(exp)

    # stimuli
    stimulus = expyriment.stimuli.Rectangle((1000, 1000))
    blank = expyriment.stimuli.BlankScreen(colour=gabor1._background_colour)

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
        cross2.present()
        exp.clock.wait(200)
        t = gabor1.present(clear=False, update=False)
        t += cross3.present(clear=False, update=True)
        #exp.clock.wait(5000)
        t += cross.present()
        mean += t
        print(t, mean / k)
        exp.clock.wait(500)
        gabor1.unload()
        gabor1 = expyriment.stimuli.extras.GaborPatch(size=500, lambda_=5, theta=45, sigma=5, phase=0.25,
                                                      position=(-500, 0), contrast=0.5 * np.random.uniform(0, 1, 1))
        gabor1.preload()

    control.end()


app = QtGui.QApplication([])
w = QtGui.QPushButton('Run exp')
w.show() #showFullScreen()
w.clicked.connect(run_exp)
app.exec_()

