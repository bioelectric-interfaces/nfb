import expyriment

from expyriment import control, design, misc
import expyriment.stimuli
import expyriment.stimuli.extras
from PyQt5 import QtCore, QtGui, QtWidgets

def run_exp():
    #control.run_test_suite()
    #control.set_develop_mode(True)

    # init
    control.defaults.initialize_delay = 0
    gabor = expyriment.stimuli.extras.GaborPatch(size=1000, lambda_=50, theta=90, sigma=150, phase=0.25, position=(1000, 0))

    exp = design.Experiment(background_colour=gabor._background_colour)
    control.initialize(exp)

    # stimuli
    stimulus = expyriment.stimuli.Rectangle((1000, 1000))
    blank = expyriment.stimuli.BlankScreen()

    # stimuli preload
    stimulus.preload()
    blank.preload()
    gabor.preload()

    # main loop
    mean = 0
    for k in range(1, 1000):
        t = gabor.present()
        # exp.clock.wait(500)
        t += blank.present()
        mean += t
        print(t, mean / k)
        exp.clock.wait(500)
    control.end()


app = QtWidgets.QApplication([])
w = QtWidgets.QPushButton('Run exp')
w.show() #showFullScreen()
w.clicked.connect(run_exp)
app.exec_()

