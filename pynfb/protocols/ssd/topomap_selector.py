from PyQt4 import QtGui
from .ssd import ssd_analysis
from .topomap_canvas import TopographicMapCanvas
from .interactive_barplot import ClickableBarplot
from numpy import arange


class TopomapSelector(QtGui.QWidget):
    def __init__(self, data, pos, names=None, sampling_freq=500, **kwargs):
        super(TopomapSelector, self).__init__(**kwargs)
        layout = QtGui.QHBoxLayout()
        layout.setMargin(0)
        freqs = arange(4, 26)
        sampling_freq = sampling_freq
        self.pos = pos
        self.names = names
        major_vals, self.topographies = ssd_analysis(data, sampling_frequency=sampling_freq, freqs=freqs)
        self.topomap = TopographicMapCanvas(self.topographies[0], self.pos, names=names, width=5, height=4, dpi=100)
        self.selector = ClickableBarplot(self, freqs, major_vals, True)
        layout.addWidget(self.selector, 2)
        layout.addWidget(self.topomap, 1)
        self.setLayout(layout)

    def select_action(self):
        index = self.selector.current_index()
        self.topomap.update_figure(self.topographies[index], self.pos, names=self.names)

    def get_current_topo(self):
        return self.topographies[self.selector.current_index()]


if __name__ == '__main__':
    app = QtGui.QApplication([])

    import numpy as np
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    from pynfb.generators import ch_names
    channels_names = ch_names[:128]
    x = np.random.rand(10000, 128)
    pos = ch_names_to_2d_pos(channels_names)
    widget = TopomapSelector(x, pos, names=channels_names)
    widget.show()
    app.exec_()
