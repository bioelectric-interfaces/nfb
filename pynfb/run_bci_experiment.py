import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from pynfb.experiment import Experiment
from pynfb.serializers.xml_ import xml_file_to_params

def main():

    app = QtWidgets.QApplication(sys.argv)
    experiment = Experiment(app, xml_file_to_params('bci_default.xml'))
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
