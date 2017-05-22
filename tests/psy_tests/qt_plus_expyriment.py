import expyriment

from expyriment import control, design, misc
import expyriment.stimuli
from PyQt4 import QtGui, QtCore

def run_exp():
    control.run_test_suite()
    control.set_develop_mode(True)
    control.defaults.initialize_delay = 0
    garbor = expyriment.stimuli.Rectangle((100, 100)) # Picture('image.png')
    cross = expyriment.stimuli.Rectangle((100, 100), position=(100, 100))
    exp = design.Experiment(background_colour=(0, 0, 0))
    control.initialize(exp)


    garbor.preload()
    time = 0
    mean = 0
    wait_time = 0
    for k in range(1, 1000):
        t = garbor.present()
        cross.present(clear=False)
        exp.clock.wait(500)
        #expyriment.stimuli.Rectangle((100, 100)).present(clear=False)
        t += expyriment.stimuli.BlankScreen().present()
        #
        mean += t
        print(t, mean / k)
        exp.clock.wait(500)
    control.end()


app = QtGui.QApplication([])
w = QtGui.QPushButton('Run exp')
w.show() #showFullScreen()
w.clicked.connect(run_exp)
app.exec_()

