from PyQt4 import QtGui
from pynfb.protocols.ssd.ssd import ssd_analysis
from pynfb.protocols.ssd.sliders import Sliders
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from pynfb.protocols.ssd.interactive_barplot import ClickableBarplot

from pynfb.widgets.parameter_slider import ParameterSlider
from numpy import arange



class TopomapSelector(QtGui.QWidget):
    def __init__(self, data, pos, names=None, sampling_freq=500, **kwargs):
        super(TopomapSelector, self).__init__(**kwargs)
        layout = QtGui.QHBoxLayout()
        layout.setMargin(0)
        v_layout = QtGui.QVBoxLayout()
        v_layout.addLayout(layout)
        self.setLayout(v_layout)

        # central bandwidth slider
        self.sliders = Sliders()
        self.sliders.apply_button.clicked.connect(self.recompute_ssd)
        v_layout.addWidget(self.sliders)
        self.x_left = 4
        self.x_right = 40
        self.x_delta = 1

        self.freqs = arange(self.x_left, self.x_right, self.x_delta)
        sampling_freq = sampling_freq
        self.pos = pos
        self.names = names
        self.data = data
        self.sampling_freq = sampling_freq
        self.major_vals, self.topographies = ssd_analysis(data, sampling_frequency=sampling_freq, freqs=self.freqs)
        self.topomap = TopographicMapCanvas(self.topographies[0], self.pos, names=names, width=5, height=4, dpi=100)
        self.selector = ClickableBarplot(self, self.freqs, self.major_vals)
        self.recompute_ssd()
        #self.selector.reset_y(np.array(major_vals)*0+1)
        #self.selector.reset_w(3)
        layout.addWidget(self.selector, 2)
        layout.addWidget(self.topomap, 1)



    def select_action(self):
        index = self.selector.current_index()
        self.topomap.update_figure(self.topographies[index], self.pos, names=self.names)

    def get_current_topo(self):
        return self.topographies[self.selector.current_index()]

    def recompute_ssd(self):
        parameters = self.sliders.getValues()
        self.freqs = arange(self.x_left, self.x_right, parameters['bandwidth'])
        self.major_vals, self.topographies = ssd_analysis(self.data,
                                                          sampling_frequency=self.sampling_freq,
                                                          freqs=self.freqs,
                                                          regularization_coef=parameters['regularizator'],
                                                          flanker_delta=parameters['flanker_bandwidth'],
                                                          flanker_margin=parameters['flanker_margin'])
        self.selector.plot(self.freqs, self.major_vals)
        self.select_action()


if __name__ == '__main__':
    app = QtGui.QApplication([])

    import numpy as np
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
                'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    channels_names = np.array(ch_names)
    x = np.loadtxt('example_recordings.txt')[:, channels_names!='Cz']
    channels_names = list(channels_names[channels_names!='Cz'])

    print(x.shape, channels_names)
    pos = ch_names_to_2d_pos(channels_names)
    widget = TopomapSelector(x, pos, names=channels_names, sampling_freq=1000)
    widget.show()
    app.exec_()
