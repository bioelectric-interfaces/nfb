from PyQt4 import QtGui
from pynfb.protocols.ssd.ssd import ssd_analysis
from pynfb.protocols.ssd.sliders_csp import Sliders
from pynfb.protocols.ssd.topomap_canvas import TopographicMapCanvas
from pynfb.protocols.ssd.interactive_barplot import ClickableBarplot

from pynfb.widgets.parameter_slider import ParameterSlider
from numpy import arange, dot, array, eye
from numpy.linalg import pinv
import numpy as np



class TopomapSelector(QtGui.QWidget):
    def __init__(self, data, pos, names, sampling_freq=500, **kwargs):
        super(TopomapSelector, self).__init__(**kwargs)

        # layouts
        layout = QtGui.QHBoxLayout()
        layout.setMargin(0)
        v_layout = QtGui.QVBoxLayout()
        v_layout.addLayout(layout)
        self.setLayout(v_layout)

        # Sliders
        self.sliders = Sliders()
        self.sliders.apply_button.clicked.connect(self.recompute)
        v_layout.addWidget(self.sliders)

        # csp properetires
        self.bandpass = (0, sampling_freq/2)
        self.pos = pos
        self.names = names
        self.data = data
        self.sampling_freq = sampling_freq

        # topomap canvas layout
        topo_layout = QtGui.QVBoxLayout()
        layout.addLayout(topo_layout, 1)

        # topomap canvas
        self.topomaps = [TopographicMapCanvas(width=5, height=4, dpi=100) for _k in range(len(names))]
        for topomap in self.topomaps:
            topo_layout.addWidget(topomap)
            topomap.setHidden(True)
        self.current_topomap = 0
        self.topomap = self.topomaps[self.current_topomap]
        self.topomap.setHidden(False)
        self.topomap_drawn = [False for topomap in self.topomaps]

        # selector barplot init
        self.selector = ClickableBarplot(self, xlabel='Eigenvalues')
        layout.addWidget(self.selector, 2)
        # self.selector.changed.connect(self.change_topomap)

        # first ssd analysis
        self.recompute()

    def change_topomap(self):
        self.topomap.setHidden(True)
        self.current_topomap = self.selector.current_index()
        self.topomap = self.topomaps[self.current_topomap]
        self.topomap.setHidden(False)
        self.draw_topomap()

    def select_action(self):
        self.change_topomap()

    def draw_topomap(self):
        if not self.topomap_drawn[self.current_topomap]:
            self.topomap.update_figure(self.topographies[:, self.current_topomap], self.pos, names=self.names)
            self.topomap_drawn[self.current_topomap] = True


    def update_data(self, data):
        self.data = data
        self.recompute()

    def recompute(self):
        self.topomap_drawn = [False for _topomap in self.topomaps]
        current = self.selector.current_index()
        if current is None:
            current = 0
        parameters = self.sliders.getValues()
        self.bandpass = (parameters['bandpass_low'], parameters['bandpass_high'])
        from pynfb.protocols.ssd.csp import csp
        self.major_vals, self.filters, self.topographies = csp(self.data,
                                                               fs=self.sampling_freq,
                                                               band=self.bandpass,
                                                               regularization_coef=parameters['regularizator'])
        self.selector.plot(np.arange(self.data.shape[1])+0.5, self.major_vals)
        self.selector.set_current(current)
        self.change_topomap()

    def get_current_filter(self, reject=False):
        filters = self.filters
        filter = filters[:, self.current_topomap]
        if reject:
            # rejected_matrix = dot(filters, eye(filters.shape[0]) - dot(filter[:, None], filter[None, :]))
            # inv = pinv(filters)
            # return dot(rejected_matrix, inv)
            inv = pinv(filters)
            filters[:, self.current_topomap] = 0
            return dot(filters, inv)
        return filter

    def get_current_bandpass(self):
        return self.bandpass


if __name__ == '__main__':
    app = QtGui.QApplication([])

    import numpy as np
    from pynfb.widgets.helpers import ch_names_to_2d_pos
    # ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
    #             'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    # channels_names = np.array(ch_names)
    # x = np.loadtxt('example_recordings.txt')[:, channels_names!='Cz']
    # channels_names = list(channels_names[channels_names!='Cz'])
    # x = np.random.randn(10000, len(channels_names))


    channels_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9', 'Fc5', 'Fc1', 'Fc2', 'Fc6', 'Ft10', 'T7', 'C3',
                      'Cz',
                      'C4', 'T8', 'Tp9', 'Cp5', 'Cp1', 'Cp2', 'Cp6', 'Tp10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz',
                      'O2']
    from pynfb.io.hdf5 import load_h5py

    x = load_h5py('C:\\Users\\Nikolai\Downloads\\raw_.h5', 'protocol1')
    y = load_h5py('C:\\Users\\Nikolai\Downloads\\raw_.h5', 'protocol2')
    x = np.vstack((x[:y.shape[0]], y))

    print(x.shape, channels_names)
    pos = ch_names_to_2d_pos(channels_names)
    widget = TopomapSelector(x, pos, names=channels_names, sampling_freq=1000)
    widget.show()
    app.exec_()
