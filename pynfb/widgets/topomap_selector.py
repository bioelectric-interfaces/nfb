from PyQt4 import QtGui, QtCore
import numpy as np
from pynfb.widgets.interactive_barplot import ClickableBarplot
from pynfb.widgets.topomap_canvas import TopographicMapCanvas
from pynfb.widgets.ssd import ssd_analysis
from pynfb.widgets.helpers import ch_names_to_2d_pos
from pynfb.generators import ch_names


class TopomapSelector(QtGui.QWidget):
    def __init__(self, data, pos, names=None, sampling_freq=500, **kwargs):
        super(TopomapSelector, self).__init__(**kwargs)
        layout = QtGui.QHBoxLayout()
        freqs = np.arange(4, 26)
        sampling_freq = sampling_freq
        self.pos = pos
        self.names = names
        major_vals, self.topographies = ssd_analysis(data, sampling_frequency=sampling_freq, freqs=freqs, flanker_delta=3)
        self.topomap = TopographicMapCanvas(self.topographies[0], self.pos, names=names, width=5, height=4, dpi=100)
        self.selector = ClickableBarplot(self, freqs, major_vals, True)
        layout.addWidget(self.selector, 2)
        layout.addWidget(self.topomap, 1)
        self.setLayout(layout)

    def select_action(self):
        index = self.selector.current_index()
        self.topomap.update_figure(self.topographies[index], self.pos, names=self.names)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    channels_names = ch_names[:90]
    x = np.random.rand(10000, 90)
    pos = ch_names_to_2d_pos(channels_names)
    widget = TopomapSelector(x, pos, names=channels_names)
    widget.show()
    app.exec_()
