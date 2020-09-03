from PyQt5 import QtCore, QtGui, QtWidgets

from pynfb.inlets.montage import Montage, azimuthal_equidistant_projection
import numpy as np

DEBUG = False


def seems_to_come_from_neuromag(list_of_ch_names):
    return list_of_ch_names[0].startswith('MEG')


def ch_names_to_2d_pos(list_of_ch_names, kind='standard_1005', azimuthal=True):
    montage = Montage(list_of_ch_names)
    pos = montage.get_pos()
    return pos


class WaitMessage(QtWidgets.QWidget):
    def __init__(self, text=''):
        super(WaitMessage, self).__init__()
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        msg = QtWidgets.QLabel(text or 'Please wait ...')
        layout.addWidget(msg)
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        self.resize(200, 100)

    def show_and_return(self):
        self.show()
        self.repaint()
        return self
