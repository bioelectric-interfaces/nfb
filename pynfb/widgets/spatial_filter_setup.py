from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from ..protocols.ssd.topomap_canvas import TopographicMapCanvas


class Table(QtWidgets.QTableWidget):
    def __init__(self, ch_names, *args):
        super(Table, self).__init__(*args)

        # set size and names
        self.setColumnCount(2)
        self.setRowCount(len(ch_names))
        self.setHorizontalHeaderLabels(['Channel', 'Weight'])

        # set ch names
        for ind, name in enumerate(ch_names):
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.setItem(ind, 0, name_item)

        # set spin boxes and default weights
        self.weights = [0 for _j in range(self.rowCount())]
        for ind, w in enumerate(self.weights):
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setValue(w)
            spin_box.setSingleStep(0.5)
            spin_box.setRange(-1e5, 1e5)
            self.setCellWidget(ind, 1, spin_box)

        # formatting
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def set_weights(self, weights):
        self.weights = weights
        for ind, w in enumerate(weights):
            self.cellWidget(ind, 1).setValue(w)

    def revert_changes(self):
        self.set_weights(self.weights)

    def commit_changes(self):
        self.weights = [self.cellWidget(ind, 1).value() for ind in range(self.rowCount())]

    def get_weights(self):
        return self.weights


class SpatialFilterSetup(QtWidgets.QDialog):
    def __init__(self, ch_names, weights=None, message=None, title='Spatial filter', **kwargs):
        super(SpatialFilterSetup, self).__init__(**kwargs)
        #
        self.ch_names = ch_names
        self.weights = weights if weights is not None else [0. for _j in self.ch_names]

        # title
        self.setWindowTitle(title)

        # layout
        layout = QtWidgets.QGridLayout(self)

        # table
        self.table = Table(ch_names)
        if weights is not None:
            self.table.set_weights(weights)
        layout.addWidget(self.table, 0, 1)

        # topomap canvas
        self.topomap = TopographicMapCanvas()
        self.topomap.update_figure(self.weights, names=ch_names, show_names=ch_names)
        layout.addWidget(self.topomap, 0, 0)

        # buttons
        btn_layout = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton('Apply')
        apply_btn.clicked.connect(self.apply_action)
        revert_btn = QtWidgets.QPushButton('Revert')
        revert_btn.clicked.connect(self.table.revert_changes)
        ok_btn = QtWidgets.QPushButton('OK')
        ok_btn.clicked.connect(self.ok_action)
        zero_btn = QtWidgets.QPushButton('Set zeros')
        zero_btn.clicked.connect(self.set_zeros)
        btn_layout.addWidget(zero_btn)
        btn_layout.addWidget(revert_btn)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout, 1, 1)

        if message is not None:
            layout.addWidget(QtWidgets.QLabel(message), 1, 0)

    def set_zeros(self):
        self.weights = [0 for _w in self.weights]
        self.table.set_weights(self.weights)
        self.apply_action()

    def apply_action(self):
        self.table.commit_changes()
        self.topomap.update_figure(self.table.get_weights(), names=self.ch_names, show_names=self.ch_names)

    def ok_action(self):
        self.table.commit_changes()
        self.weights = self.table.get_weights()
        self.close()

    @staticmethod
    def get_filter(ch_names, **kwargs):
        selector = SpatialFilterSetup(ch_names, **kwargs)
        _result = selector.exec_()
        return selector.weights


if __name__ == '__main__':
    import numpy as np

    app = QtWidgets.QApplication([])
    ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
                'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    w = SpatialFilterSetup.get_filter(ch_names, np.random.uniform(size=len(ch_names)), message='Current spatial filter '
                                                                                               'for signal is null vector. '
                                                               'Please modify it.')
    print(w)
