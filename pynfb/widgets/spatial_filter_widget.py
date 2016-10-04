from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from PyQt4 import QtGui, QtCore

class SpatialFilterWidget(QtGui.QWidget):
    def __init__(self, parent=None, channel_names=None):
        self.channel_names = channel_names or []
        super(SpatialFilterWidget, self).__init__(parent)
        self.setMaximumHeight(100)
        layout = QtGui.QHBoxLayout(self)
        layout.setMargin(0)

        # canvas
        canvas = TopographicMapCanvas()
        canvas.draw_central_text("not\nfound", right_bottom_text='Wow')
        layout.addWidget(canvas)

        # buttons
        buttons_layout = QtGui.QVBoxLayout()
        layout.addLayout(buttons_layout)
        self.edit_button = QtGui.QPushButton('Edit')
        self.load_button = QtGui.QPushButton('Load')
        # buttons_layout.addWidget(self.edit_button)
        buttons_layout.addWidget(self.load_button)

        # handlers
        self.edit_button.clicked.connect(lambda: self.handle_edit())
        self.load_button.clicked.connect(lambda: self.handle_load())

    def handle_edit(self):
        print('edit')

    def handle_load(self):
        print('load')


if __name__ == '__main__':
    a = QtGui.QApplication([])
    w = SpatialFilterWidget()
    w.show()
    a.exec_()
