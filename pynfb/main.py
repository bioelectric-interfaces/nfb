import sys
import os
import matplotlib
matplotlib.use('TkAgg')
full_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__))+'/..')
#from pynfb import STATIC_PATH
STATIC_PATH = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static') #TODO
print(full_path)
sys.path.insert(0, full_path)
from pynfb.settings_widget import SettingsWidget
from PyQt4 import QtGui
import sys
from pynfb.io.xml_ import *


class TheMainWindow(QtGui.QMainWindow):

    def __init__(self, app):
        super(TheMainWindow, self).__init__()
        self.setWindowIcon(QtGui.QIcon(STATIC_PATH + '/imag/settings.png'))
        self.app = app
        self.initUI()


    def initUI(self):
        # exit action
        exitAction = QtGui.QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        # open file action
        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.open_event)
        # save file action
        saveFile = QtGui.QAction(QtGui.QIcon('save.png'), 'Save as..', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.setStatusTip('Save settings file')
        saveFile.triggered.connect(self.save_event)
        # status bar init
        self.statusBar()
        # menu bar init
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu.addAction(exitAction)
        # parameter tree
        self.widget = SettingsWidget(self.app)
        self.setCentralWidget(self.widget)
        # window settings
        self.setGeometry(200, 200, 500, 400)
        self.setWindowTitle('Experiment settings')
        self.show()

    def open_event(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        params = xml_file_to_params(fname)
        self.widget.params = params
        self.widget.reset_parameters()

    def save_event(self):
        #print(self.widget.params)
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', './')
        #print(self.widget.params)
        params_to_xml_file(self.widget.params, fname)




def main():

    app = QtGui.QApplication(sys.argv)
    ex = TheMainWindow(app)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()