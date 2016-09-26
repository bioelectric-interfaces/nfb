from PyQt4 import QtGui, QtCore


class RejectionIcon(QtGui.QLabel):
    def __init__(self, rank=1, type_str='ICA'):
        super(RejectionIcon, self).__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText('{}\n rank = {} '.format(type_str, rank))


class RejectionsWidget(QtGui.QTableWidget):
    rejection_deleted = QtCore.pyqtSignal(int)

    def __init__(self):
        super(RejectionsWidget, self).__init__()
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.verticalHeader().setStretchLastSection(True)
        self.setRowCount(1)
        self.rejections = []

    def add_item(self, rank, type_str):
        self.rejections.append((rank, type_str))
        self.setColumnCount(self.columnCount() + 1)
        self.setCellWidget(0, self.columnCount() - 1, RejectionIcon(rank, type_str))
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def set_rejections(self, rejections):
        for rejection in rejections.list:
            self.add_item(rejection.rank, rejection.type_str)

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


if __name__ == '__main__':
    a = QtGui.QApplication([])

    import numpy as np
    w = RejectionsWidget()

    w.show()
    w.add_item(1, 'ICA')
    w.add_item(3, 'CSP')
    w.add_item(3, 'CSP')
    w.add_item(4, 'CSP')
    w.rejection_deleted.connect(lambda x: print(x, 'deleted'))
    a.exec_()