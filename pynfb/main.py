from pynfb.settings_widget import SettingsWidget
from pynfb.settings_widget import parameters as p
from PyQt4 import QtGui, QtCore

import sys
#from pyqtgraph.parametertree import ParameterTree, Parameter
from pynfb.experiment_parameters.widgets import *
from pynfb.experiment import Experiment


class TheMainWindow(QtGui.QMainWindow):

    def __init__(self, app):
        super(TheMainWindow, self).__init__()
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
        openFile.triggered.connect(self.showDialog)
        # save file action
        saveFile = QtGui.QAction(QtGui.QIcon('save.png'), 'Save', self)
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
        self.widget = SettingsWidget()
        self.setCentralWidget(self.widget)
        # window settings
        self.setGeometry(200, 200, 500, 400)
        self.setWindowTitle('Experiment settings')
        self.show()

    def showDialog(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        params = xml_file_to_params(fname)
        self.widget.params = params
        self.widget.reset_parameters()
        #params = formatted_odict_to_params(format_odict_by_defaults(odict, general_defaults))
        #params += vector_formatted_odict_to_params(format_odict_by_defaults(odict, vectors_defaults))
        #self.widget.parameter_object = Parameter.create(name='params', type='group', children=params)
        #self.widget.tree.setParameters(self.widget.parameter_object, showTop=False)

    def save_event(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', './')
        params_to_xml_file(self.widget.params, fname)
            # params = formatted_odict_to_params(format_odict_by_defaults(odict, general_defaults))
            # params += vector_formatted_odict_to_params(format_odict_by_defaults(odict, vectors_defaults))
            # self.widget.parameter_object = Parameter.create(name='params', type='group', children=params)
            # self.widget.tree.setParameters(self.widget.parameter_object, showTop=False)


class Widget(QtGui.QWidget):
    def __init__(self, app):
        super(Widget, self).__init__()
        self.app = app
        self.initUI()

    def initUI(self):
        styleSheet = """
            QTreeView::item {
                border: 1px solid #d9d9d9;
                border-top-color: transparent;
                border-bottom-color: transparent;
                border-left-color: transparent;
                border-right-color: transparent;
                color: #000000
            }
            """
        self.setStyleSheet(styleSheet)
        layout = QtGui.QGridLayout()
        tree = ParameterTree()
        layout.addWidget(tree)
        start_button = QtGui.QPushButton('Start')
        start_button.clicked.connect(self.onClicked)
        layout.addWidget(start_button)
        self.setLayout(layout)
        #odict = read_xml_to_dict('pynfb/experiment_parameters/settings/pilot.xml', True)
        params = formatted_odict_to_params(general_defaults)
        params += vector_formatted_odict_to_params(vectors_defaults)
        # Create tree of Parameter objects
        self.parameter_object = Parameter.create(name='params', type='group', children=params)
        # Create two ParameterTree widgets, both accessing the same data
        tree.setParameters(self.parameter_object, showTop=False)
        self.tree = tree

    def onClicked(self):
        params = params_to_odict(self.parameter_object.getValues())
        self.experiment = Experiment(self.app, params)

def main():

    app = QtGui.QApplication(sys.argv)
    ex = TheMainWindow(app)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()