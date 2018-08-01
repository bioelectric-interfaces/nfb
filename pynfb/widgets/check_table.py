from PyQt5 import QtCore, QtGui, QtWidgets

class CheckTable(QtWidgets.QTableWidget):
    def __init__(self, names, state_names, name_col, *args):
        super(CheckTable, self).__init__(*args)

        # attributes
        self.row_items_max_height = 125
        self.names = names
        self.order = list(range(len(self.names)))
        self.n_check_rows = len(state_names)

        # set size and names
        self.columns = state_names + [name_col]
        self.col = 0
        self.setColumnCount(len(self.columns))
        self.setRowCount(len(self.names))
        self.setHorizontalHeaderLabels(self.columns)

        # columns widgets
        self.checkboxes = []
        for ind in range(self.rowCount()):
            # checkboxes
            checkboxes = []
            for j in range(self.n_check_rows):
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(j == 0)
                checkboxes.append(checkbox)
                self.setCellWidget(ind, j, checkbox)
            self.checkboxes.append(checkboxes)

            # name
            self.setCellWidget(ind, self.columns.index(name_col), QtWidgets.QLabel(self.names[ind]))

        # formatting
        self.current_row = None
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # selection context menu
        header = self.horizontalHeader()
        header.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.handle_header_menu)

        # ctrl+a short cut
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_A), self).activated.connect(self.ctrl_plus_a_event)

        # checkbox cell clicked
        self.cellClicked.connect(self.cell_was_clicked)

    def cell_was_clicked(self, row, column):
        if column < self.n_check_rows:
            self.checkboxes[self.order[row]][column].click()

    def ctrl_plus_a_event(self):
        if len(self.get_unchecked_rows()) == 0:
            self.clear_selection()
        else:
            self.select_all()

    def contextMenuEvent(self, pos):
        self.checkboxes[self.order[self.rowAt(pos.y())]][self.columnAt(pos.x())].click() # state will be unchanged
        self.open_selection_menu(self.columnAt(pos.x()))

    def handle_header_menu(self, pos):
        if self.horizontalHeader().logicalIndexAt(pos) in range(self.n_check_rows):
            self.open_selection_menu(self.columnAt(pos.x()))

    def open_selection_menu(self, col):
        self.col = col
        menu = QtWidgets.QMenu()
        for name, method in zip(['Revert selection', 'Select all', 'Clear selection'],
                                [self.revert_selection, self.select_all, self.clear_selection]):
            action = QtWidgets.QAction(name, self)
            action.triggered.connect(method)
            menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    def revert_selection(self):
        col = self.col
        for checkbox in self.checkboxes:
            checkbox[col].setChecked(not checkbox[col].isChecked())

    def select_all(self):
        col = self.col
        for checkbox in self.checkboxes:
            checkbox[col].setChecked(True)

    def clear_selection(self):
        col = self.col
        for checkbox in self.checkboxes:
            checkbox[col].setChecked(False)

    def get_checked_rows(self):
        return [[jj for jj, checkbox in enumerate(self.checkboxes) if checkbox[j].isChecked()]
                for j in range(self.n_check_rows)]

    def get_unchecked_rows(self):
        return [[jj for jj, checkbox in enumerate(self.checkboxes) if not checkbox[j].isChecked()]
                for j in range(self.n_check_rows)]


if __name__ == '__main__':
    a = QtWidgets.QApplication([])
    w = CheckTable(['One', 'Two'], ['State1', 'saef', 'saeg'], 'weg')
    w.show()
    a.exec_()
    print(w.get_checked_rows())
