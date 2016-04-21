from lsltools import sim, vis
import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True)
from seaborn import color_palette
from matplotlib.colors import rgb2hex
cm = [rgb2hex(c)[1:] for c in color_palette()]

# STEP 1: Initialize a generator for simulated EEG and start it up.
#eeg_data = sim.EEGData(nch=3,stream_name="example")
eeg_data = sim.EEGData(nch=20, stream_name="example", srate=1)
eeg_data.start()

# STEP 2: Find the stream started in step 1 and pass it to the vis.Grapher
#streams = vis.pylsl.resolve_byprop("name","example")
#eeg_graph = vis.Grapher(streams[0], 512*30, cm[0])
