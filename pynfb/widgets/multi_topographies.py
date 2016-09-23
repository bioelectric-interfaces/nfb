from PyQt4 import QtGui

from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas


class MultiTopographiesCanvas(QtGui.QTableWidget):
    def __init__(self, pos, names):
        super(MultiTopographiesCanvas, self).__init__()
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setRowCount(1)
        self.pos = pos
        self.names = names

    def add_topography(self, topography=None):
        topography_widget = TopographicMapCanvas(self)
        size = 50
        topography_widget.setMaximumWidth(size)
        topography_widget.setMaximumHeight(size)
        topography_widget.setMinimumWidth(size)
        topography_widget.setMinimumHeight(size)

        if topography is not None:
            topography_widget.update_figure(topography, self.pos, self.names, show_names=[], show_colorbar=False)
        else:
            topography_widget.draw_central_text(show_not_found_symbol=True)
        self.setColumnCount(self.columnCount() + 1)
        self.setCellWidget(0, self.columnCount() - 1, topography_widget)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def contextMenuEvent(self, pos):
        self.open_selection_menu(self.columnAt(pos.x()), self.rowAt(pos.y()))

    def open_selection_menu(self, column, row):
        menu = QtGui.QMenu()
        action = QtGui.QAction('Delete', self)
        action.triggered.connect(lambda: self.delete_rejection(column, row))
        menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    def delete_rejection(self, column, row):
        print(column, row, 'deleted')


a = QtGui.QApplication([])

import numpy as np
w = MultiTopographiesCanvas(pos=np.array([(0, 0), (0, 1), (1, -1)]), names=['Cp', 'Cz', 'Fp1'])

w.show()
w.add_topography(np.array([2, 0, 0]))
w.add_topography(np.array([0, 0, 2]))
w.add_topography(None)
a.exec_()