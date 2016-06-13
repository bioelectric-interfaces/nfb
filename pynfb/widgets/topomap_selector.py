from PyQt4 import QtGui, QtCore
import numpy as np
from pynfb.widgets.interactive_barplot import ClickableBarplot
from pynfb.widgets.topomap_canvas import TopographicMapCanvas


class TopomapSelector(QtGui.QWidget):
    def __init__(self, **kwargs):
        super(TopomapSelector, self).__init__(**kwargs)
        layout = QtGui.QHBoxLayout()
        self.topomap = TopographicMapCanvas(np.random.randn(3), np.array([(0, 0), (1, -1), (-1, -1)]), width=5, height=4, dpi=100)
        self.selector = ClickableBarplot(self, np.linspace(0, 1, 50), np.random.uniform(size=50) + np.sin(np.arange(50) / 10) + 1, True)
        layout.addWidget(self.selector, 2)
        layout.addWidget(self.topomap, 1)
        self.setLayout(layout)

    def select_action(self):
        self.topomap.test_update_figure()



if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = TopomapSelector()
    widget.show()
    app.exec_()
