from PyQt5 import QtGui, QtWidgets

from ..protocols.ssd.topomap_canvas import TopographicMapCanvas


class MultiTopographiesCanvas(QtWidgets.QTableWidget):
    def __init__(self, names):
        super(MultiTopographiesCanvas, self).__init__()
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setRowCount(1)
        self.names = names

    def add_topography(self, topography=None):
        topography_widget = TopographicMapCanvas(self)
        size = 80
        topography_widget.setMaximumWidth(size)
        topography_widget.setMaximumHeight(size)
        topography_widget.setMinimumWidth(size)
        topography_widget.setMinimumHeight(size)

        if topography is not None:
            topography_widget.update_figure(topography, names=self.names, show_names=[], show_colorbar=False)
        else:
            topography_widget.draw_central_text(show_not_found_symbol=True)
        self.setColumnCount(self.columnCount() + 1)
        self.setCellWidget(0, self.columnCount() - 1, topography_widget)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


class TopographiesDialog(QtWidgets.QDialog):
    def __init__(self, names, title='', parent=None):
        super(TopographiesDialog, self).__init__(parent)
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        self.table = MultiTopographiesCanvas(names)
        layout.addWidget(self.table)
        self.resize(400, 100)

if __name__ == '__main__':
    a = QtWidgets.QApplication([])

    import numpy as np
    w = TopographiesDialog(names=['Cp1', 'Cz', 'Fp1'])

    w.show()
    w.table.add_topography(np.array([2, 0, 0]))
    w.table.add_topography(np.array([0, 0, 2]))
    w.table.add_topography(None)
    a.exec_()
