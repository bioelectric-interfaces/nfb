import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.phonon import Phonon
app = QtWidgets.QApplication(sys.argv)
vp = Phonon.VideoPlayer()

vp.show()
media = Phonon.MediaSource('drop.avi')
vp.load(media)
vp.play()
sys.exit(app.exec_())
