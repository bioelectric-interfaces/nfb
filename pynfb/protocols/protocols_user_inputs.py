from pynfb.widgets.topomap_selector import TopomapSelector
from pynfb.generators import ch_names
from pynfb.widgets.helpers import ch_names_to_2d_pos
from PyQt4 import QtGui, QtCore
import numpy as np


class SelectSSDFilterWidget(QtGui.QDialog):
    def __init__(self, data, pos, names=None, sampling_freq=500, parent=None):
        super(SelectSSDFilterWidget, self).__init__(parent)
        # layout
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        # label
        top_label = QtGui.QLabel('Select filter:')
        layout.addWidget(top_label)

        # topomap selector
        self.selector = TopomapSelector(data, pos, names=names, sampling_freq=sampling_freq)
        layout.addWidget(self.selector)

        # select button
        select_button = QtGui.QPushButton('Select')
        select_button.setMaximumWidth(100)
        select_button.clicked.connect(self.select_action)
        layout.addWidget(select_button)

        # selected filter
        self.filter = self.selector.get_current_topo()

    def select_action(self):
        self.filter = self.selector.get_current_topo()
        self.close()
        # print(self.filter)


    @staticmethod
    def select_filter(data, pos, names=None, sampling_freq=500, parent=None):
        selector = SelectSSDFilterWidget(data, pos, names=names, sampling_freq=sampling_freq, parent=parent)
        result = selector.exec_()
        print(selector.filter)
        return selector.filter


if __name__ == '__main__':
    app = QtGui.QApplication([])
    channels_names = ch_names[:128]
    x = np.random.rand(10000, 128)
    pos = ch_names_to_2d_pos(channels_names)
    widget = SelectSSDFilterWidget(x, pos, names=channels_names)
    widget.show()
    app.exec_()