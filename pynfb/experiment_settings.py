import sys
from PyQt4 import QtGui
from pyqtgraph.parametertree import ParameterTree, Parameter
from pynfb.experiment_parameters.widgets import *


class Example(QtGui.QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

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
        # status bar init
        self.statusBar()
        # menu bar init
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(exitAction)
        # parameter tree
        self.widget = Widget()
        self.setCentralWidget(self.widget)
        # window settings
        self.setGeometry(200, 200, 500, 400)
        self.setWindowTitle('Experiment settings')
        self.show()

    def showDialog(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        odict = read_xml_to_dict(fname, True)
        params = formatted_odict_to_params(format_odict_by_defaults(odict, general_defaults))
        params += vector_formatted_odict_to_params(format_odict_by_defaults(odict, vectors_defaults))
        p = Parameter.create(name='params', type='group', children=params)
        self.widget.t.setParameters(p, showTop=False)


class Widget(QtGui.QWidget):
    def __init__(self):
        super(Widget, self).__init__()

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
        t = ParameterTree()
        layout.addWidget(t)
        start_button = QtGui.QPushButton('Start')
        layout.addWidget(start_button)
        self.setLayout(layout)
        #odict = read_xml_to_dict('pynfb/experiment_parameters/settings/pilot.xml', True)
        params = formatted_odict_to_params(general_defaults)
        params += vector_formatted_odict_to_params(vectors_defaults)
        # Create tree of Parameter objects
        p = Parameter.create(name='params', type='group', children=params)
        # Create two ParameterTree widgets, both accessing the same data
        t.setParameters(p, showTop=False)
        self.t = t

def main():

    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()