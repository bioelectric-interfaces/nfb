from PyQt4 import QtGui, QtCore

class CheckTable(QtGui.QTableWidget):
    one_selected = QtCore.pyqtSignal()
    more_one_selected = QtCore.pyqtSignal()
    no_one_selected = QtCore.pyqtSignal()

    def __init__(self, names, *args):
        super(CheckTable, self).__init__(*args)

        # attributes
        self.row_items_max_height = 125
        self.names = names
        self.order = list(range(len(self.names)))

        # set size and names
        self.columns = ['Selection', 'Name']
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(self.names))
        self.setHorizontalHeaderLabels(self.columns)

        # columns widgets
        self.checkboxes = []
        for ind in range(self.rowCount()):
            # checkboxes
            checkbox = QtGui.QCheckBox()
            checkbox.setChecked(True)
            self.checkboxes.append(checkbox)
            self.setCellWidget(ind, self.columns.index('Selection'), checkbox)

            # name
            self.setCellWidget(ind, self.columns.index('Name'), QtGui.QLabel(self.names[ind]))

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # checkbox signals
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.checkboxes_state_changed)

        # selection context menu
        header = self.horizontalHeader()
        header.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.handle_header_menu)

        # ctrl+a short cut
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_A), self),
                     QtCore.SIGNAL('activated()'), self.ctrl_plus_a_event)

        # checkbox cell clicked
        self.cellClicked.connect(self.cell_was_clicked)

    def cell_was_clicked(self, row, column):
        if column == 0:
            self.checkboxes[self.order[row]].click()

    def ctrl_plus_a_event(self):
        if len(self.get_unchecked_rows()) == 0:
            self.clear_selection()
        else:
            self.select_all()

    def contextMenuEvent(self, pos):
        if self.columnAt(pos.x()) == 0:
            self.checkboxes[self.order[self.rowAt(pos.y())]].click() # state will be unchanged
            self.open_selection_menu()

    def handle_header_menu(self, pos):
        if self.horizontalHeader().logicalIndexAt(pos) == 0:
            self.open_selection_menu()

    def open_selection_menu(self):
        menu = QtGui.QMenu()
        for name, method in zip(['Revert selection', 'Select all', 'Clear selection'],
                                [self.revert_selection, self.select_all, self.clear_selection]):
            action = QtGui.QAction(name, self)
            action.triggered.connect(method)
            menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    def revert_selection(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(not checkbox.isChecked())

    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def clear_selection(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def checkboxes_state_changed(self):
        checked = self.get_checked_rows()
        if len(checked) == 0:
            self.no_one_selected.emit()
        elif len(checked) == 1:
            self.one_selected.emit()
        else:
            self.more_one_selected.emit()

    def get_checked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

    def get_unchecked_rows(self):
        return [j for j, checkbox in enumerate(self.checkboxes) if not checkbox.isChecked()]


if __name__ == '__main__':
    a = QtGui.QApplication([])
    w = CheckTable(['One', 'Two'])
    w.show()
    a.exec_()
    print(w.get_checked_rows())