from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from PyQt5 import QtCore, QtGui, QtWidgets

class SpatialFilterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, channel_names=None):
        self.channel_names = channel_names or []
        super(SpatialFilterWidget, self).__init__(parent)
        self.setMaximumHeight(100)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # canvas
        canvas = TopographicMapCanvas()
        canvas.draw_central_text("not\nfound", right_bottom_text='Wow')
        layout.addWidget(canvas)

        # buttons
        buttons_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(buttons_layout)
        self.edit_button = QtWidgets.QPushButton('Edit')
        self.load_button = QtWidgets.QPushButton('Load')
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
    a = QtWidgets.QApplication([])
    w = SpatialFilterWidget()
    w.show()
    a.exec_()
