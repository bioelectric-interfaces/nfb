from PyQt4 import QtGui, QtCore


class RejectionIcon(QtGui.QLabel):
    def __init__(self, rank=1, type_str='ICA'):
        super(RejectionIcon, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText('{}\nrank = {}'.format(type_str, rank))


class MultiTopographiesCanvas(QtGui.QTableWidget):
    rejection_deleted = QtCore.pyqtSignal(int)

    def __init__(self, pos, names):
        super(MultiTopographiesCanvas, self).__init__()
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setRowCount(1)
        self.pos = pos
        self.names = names
        self.rejections = []

    def add_item(self, rank, type_str):
        self.rejections.append((rank, type_str))
        self.setColumnCount(self.columnCount() + 1)
        self.setCellWidget(0, self.columnCount() - 1, RejectionIcon(rank, type_str))
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def contextMenuEvent(self, pos):
        self.open_selection_menu(self.columnAt(pos.x()), self.rowAt(pos.y()))

    def open_selection_menu(self, column, row):
        if row >= 0 and column >= 0:
            menu = QtGui.QMenu()
            action = QtGui.QAction('Delete', self)
            action.triggered.connect(lambda: self.delete_rejection(column, row))
            menu.addAction(action)
            menu.exec_(QtGui.QCursor.pos())

    def delete_rejection(self, column, row):
        self.rejections.pop(column)
        self.setColumnCount(len(self.rejections))
        for j, rejection in enumerate(self.rejections):
            self.setCellWidget(0, j, RejectionIcon(*rejection))
        self.rejection_deleted.emit(column)



a = QtGui.QApplication([])

import numpy as np
w = MultiTopographiesCanvas(pos=np.array([(0, 0), (0, 1), (1, -1)]), names=['Cp', 'Cz', 'Fp1'])

w.show()
w.add_item(1, 'ICA')
w.add_item(3, 'CSP')
w.add_item(3, 'CSP')
w.add_item(4, 'CSP')
w.rejection_deleted.connect(lambda x: print(x, 'deleted'))
a.exec_()