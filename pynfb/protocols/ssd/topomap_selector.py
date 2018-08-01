from PyQt5 import QtGui, QtWidgets
from ...protocols.ssd.ssd import ssd_analysis
from ...protocols.ssd.sliders import Sliders
from ...protocols.ssd.topomap_canvas import TopographicMapCanvas
from ...protocols.ssd.interactive_barplot import ClickableBarplot

from ...widgets.parameter_slider import ParameterSlider
from numpy import arange, dot, array, eye
from numpy.linalg import pinv



class TopomapSelector(QtWidgets.QWidget):
    def __init__(self, data, pos, names, sampling_freq=500, **kwargs):
        super(TopomapSelector, self).__init__(**kwargs)

        # layouts
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addLayout(layout)
        self.setLayout(v_layout)

        # Sliders
        self.sliders = Sliders()
        self.sliders.apply_button.clicked.connect(self.recompute)
        v_layout.addWidget(self.sliders)

        # ssd properetires
        self.x_left = 4
        self.x_right = 40
        self.x_delta = 1
        self.freqs = arange(self.x_left, self.x_right, self.x_delta)
        self.pos = pos
        self.names = names
        self.data = data
        self.sampling_freq = sampling_freq

        # topomap canvas layout
        topo_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(topo_layout, 1)

        # component spinbox and layout
        component_layout = QtWidgets.QHBoxLayout()
        self.component_spinbox = QtWidgets.QSpinBox()
        self.component_spinbox.setRange(1, len(names))
        self.component_spinbox.valueChanged.connect(self.change_topomap)
        self.component_spinbox.valueChanged.connect(self.draw_lambda_level)
        component_layout.addWidget(QtWidgets.QLabel('Component:'))
        component_layout.addWidget(self.component_spinbox)

        # topomap canvas
        self.topomaps = [TopographicMapCanvas(width=5, height=4, dpi=100) for _k in range(len(names))]
        for topomap in self.topomaps:
            topo_layout.addWidget(topomap)
            topomap.setHidden(True)
        self.current_topomap = 0
        self.topomap = self.topomaps[self.current_topomap]
        self.topomap.setHidden(False)
        self.topomap_drawn = [False for topomap in self.topomaps]
        topo_layout.addLayout(component_layout)

        # selector barplot init
        self.selector = ClickableBarplot(self)
        layout.addWidget(self.selector, 2)
        self.selector.changed.connect(self.underline_central_band)
        self.selector.changed.connect(self.draw_lambda_level)

        # first ssd analysis
        self.recompute()

    def change_topomap(self):
        self.topomap.setHidden(True)
        self.current_topomap = self.component_spinbox.value() - 1
        self.topomap = self.topomaps[self.current_topomap]
        self.topomap.setHidden(False)
        self.draw_topomap()

    def select_action(self):
        self.topomap_drawn = [False for _topomap in self.topomaps]
        self.draw_topomap()

    def draw_topomap(self):
        index = self.selector.current_index()
        if not self.topomap_drawn[self.current_topomap]:
            self.topomap.update_figure(self.topographies[index][:, self.current_topomap], self.pos, names=self.names)
            self.topomap_drawn[self.current_topomap] = True

    def get_current_topo(self):
        return self.topographies[self.selector.current_index()][:, self.current_topomap]

    def get_current_filter(self, reject=False):
        filters = self.filters[self.selector.current_index()]
        filter = filters[:, self.current_topomap]
        if reject:
            # rejected_matrix = dot(filters, eye(filters.shape[0]) - dot(filter[:, None], filter[None, :]))
            # inv = pinv(filters)
            # return dot(rejected_matrix, inv)
            inv = pinv(filters)
            filters[:, self.current_topomap] = 0
            return dot(filters, inv)
        return filter

    def update_data(self, data):
        self.data = data
        self.recompute()

    def get_current_bandpass(self):
        x1 = self.selector.current_x()
        x2 = x1 + self.x_delta
        return x1 - self.flanker_margin - self.flanker_delta, x2 + self.flanker_margin + self.flanker_delta

    def recompute(self):
        self.topomap_drawn = [False for _topomap in self.topomaps]
        current_x = self.selector.current_x()
        parameters = self.sliders.getValues()
        self.x_delta = parameters['bandwidth']
        self.freqs = arange(self.x_left, self.x_right, self.x_delta)
        self.flanker_delta = parameters['flanker_bandwidth']
        self.flanker_margin = parameters['flanker_margin']
        self.major_vals, self.topographies, self.filters = ssd_analysis(self.data,
                                                                        sampling_frequency=self.sampling_freq,
                                                                        freqs=self.freqs,
                                                                        regularization_coef=parameters['regularizator'],
                                                                        flanker_delta=self.flanker_delta,
                                                                        flanker_margin=self.flanker_margin)
        self.selector.plot(self.freqs, self.major_vals[:, 0])
        self.selector.set_current_by_value(current_x)
        self.change_topomap()

    def underline_central_band(self):
        self.selector.clear_underlines_and_ticks()
        x1 = self.selector.current_x()
        x2 = x1 + self.x_delta
        self.selector.underline(x1 - self.flanker_margin - self.flanker_delta, x1 - self.flanker_margin, 'flanker')
        self.selector.underline(x2 + self.flanker_margin, x2 + self.flanker_margin + self.flanker_delta, 'flanker')
        self.selector.underline(x1, x2, 'central')
        self.selector.add_xtick(x1 - self.flanker_margin - self.flanker_delta)
        self.selector.add_xtick(x2 + self.flanker_margin + self.flanker_delta)

    def draw_lambda_level(self):
        self.selector.update_bin_level(self.x_delta,
                                       self.major_vals[self.selector.current_index(), self.current_topomap])


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    import numpy as np
    from ..widgets.helpers import ch_names_to_2d_pos
    ch_names = ['Fc1', 'Fc3', 'Fc5', 'C1', 'C3', 'C5', 'Cp1', 'Cp3', 'Cp5', 'Cz', 'Pz',
                'Cp2', 'Cp4', 'Cp6', 'C2', 'C4', 'C6', 'Fc2', 'Fc4', 'Fc6']
    channels_names = np.array(ch_names)
    x = np.loadtxt('example_recordings.txt')[:, channels_names!='Cz']
    channels_names = list(channels_names[channels_names!='Cz'])
    # x = np.random.randn(10000, len(channels_names))

    print(x.shape, channels_names)
    pos = ch_names_to_2d_pos(channels_names)
    widget = TopomapSelector(x, pos, names=channels_names, sampling_freq=1000)
    widget.show()
    app.exec_()
