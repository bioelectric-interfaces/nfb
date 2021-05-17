import sys
import os
import argparse
import multiprocessing
import traceback

import pynfb
import matplotlib
matplotlib.use('TkAgg')
full_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__))+'/..')
#from pynfb import STATIC_PATH
STATIC_PATH = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static') #TODO
print(full_path)
sys.path.insert(0, full_path)
from pynfb.settings_widget import SettingsWidget
from pynfb.experiment import Experiment
from pynfb.widgets.helpers import ScrollArea
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
import sys
from pynfb.serializers.xml_ import *


class TheMainWindow(QtWidgets.QMainWindow):

    def __init__(self, app):
        super(TheMainWindow, self).__init__()
        self.setWindowIcon(QtGui.QIcon(STATIC_PATH + '/imag/settings.png'))
        self.app = app
        
        # Set custom excepthook that shows the error message as a QMessageBox
        self.__excepthook__ = sys.excepthook
        sys.excepthook = self.excepthook

        self.initUI()


    def initUI(self):
        # exit action
        exitAction = QtWidgets.QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtWidgets.QApplication.quit)
        # open file action
        openFile = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.open_event)
        # save file action
        saveFile = QtWidgets.QAction(QtGui.QIcon('save.png'), 'Save as..', self)
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
        scrollArea = ScrollArea()
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollArea.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        scrollArea.setWidgetResizable(True)
        scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        scrollArea.setWidget(self.widget)
        self.setCentralWidget(scrollArea)
        # window settings
        self.setWindowTitle('Experiment settings')
        self.show()

    def open_event(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', './')[0]

        if fname == "":
            return  # User cancelled

        params = xml_file_to_params(fname)
        self.widget.params = params
        self.widget.reset_parameters()

    def save_event(self):
        #print(self.widget.params)
        fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', './')[0]

        if fname == "":
            return  # User cancelled
        #print(self.widget.params)
        params_to_xml_file(self.widget.params, fname)

    def excepthook(self, etype, value, tb):
        message = tracebackMessageBox(self, etype, value, tb)
        message.exec_()

        self.__excepthook__(etype, value, tb)


def tracebackMessageBox(parent, etype, value, tb):
    msg = QtWidgets.QMessageBox(parent)
    msg.setWindowTitle("Critical Error")
    msg.setText(
        "An unknown critical error occured. It is recommended to save your work and restart NFB Lab.\n\n"
        "Please inform the developer, describing what you were doing before the error, and attach the text below."
    )
    msg.setIcon(msg.Critical)

    exception_field = QtWidgets.QTextEdit()
    exception_field.setText("".join(traceback.format_exception(etype, value, tb)))
    exception_field.setReadOnly(True)
    msg.layout().addWidget(exception_field, 1, 0, 1, -1)

    return msg


def run(path):
    """Run the experiment without showing config UI."""
    app = QtWidgets.QApplication(sys.argv)

    # Set up exceptions
    __excepthook__ = sys.excepthook
    def excepthook(etype, value, tb):
        message = tracebackMessageBox(None, etype, value, tb)
        message.exec_()
        __excepthook__(etype, value, tb)
    sys.excepthook = excepthook

    params = xml_file_to_params(path)
    ex = Experiment(app, params)

    return app.exec_()


def main():
    # Parse and act upon commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="open an xml experiment file when launched (optional)")
    parser.add_argument("-x", "--execute", action="store_true", help="run the experiment without configuring (requires file to be specified)")
    args = parser.parse_args()

    if args.file is None and args.execute:
        print("Could not execute the experiment without configuring because file argument was not specified")
        parser.print_help()
        sys.exit(1)

    if args.execute:
        # If "Execute" was specified, run the experiment immediately
        sys.exit(run(args.file))

    app = QtWidgets.QApplication(sys.argv)
    main_window = TheMainWindow(app)

    if args.file:
        # If "file" was specified, open the experiment file right away
        params = xml_file_to_params(args.file)
        main_window.widget.params = params
        main_window.widget.reset_parameters()

    return app.exec_()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Support running nfb in frozen mode (i.e. as an executable)
    sys.exit(main())
