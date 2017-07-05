
from __future__ import absolute_import, print_function, division
from builtins import *


import os
import tempfile
import pylab as plt

from types import ModuleType

from expyriment.stimuli._picture import Picture
from expyriment import stimuli

from expyriment.stimuli.extras import defaults

try:
    import numpy as np
except:
    np = None
try:
    from matplotlib import pyplot
except:
    pyplot = None


class ABCCross(Picture):
    """A class implementing a Gabor Patch."""

    def __init__(self, size=None, position=None, lambda_=None, theta=None,
                sigma=None, phase=None, trim=None, contrast=0.5, width=1, hide_dot=False):

        if not isinstance(np, ModuleType):
            message = """GaborPatch can not be initialized.
The Python package 'Numpy' is not installed."""
            raise ImportError(message)

        if not isinstance(pyplot, ModuleType):
            message = """GaborPatch can not be initialized.
The Python package 'Matplotlib' is not installed."""
            raise ImportError(message)

        fid, filename = tempfile.mkstemp(
                    dir=stimuli.defaults.tempdir,
                    suffix=".png")
        os.close(fid)
        Picture.__init__(self, filename, position)

        size = 200
        diameter = 50
        line_width = 20 * width
        inner_diameter = 5 * (int(hide_dot)-1)

        fig = plt.figure(frameon=False, figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        pixel_map = np.zeros((size, size))
        ax.imshow(pixel_map, cmap=plt.get_cmap('gray'))
        ax.add_patch(plt.Circle((size / 2, size / 2), diameter, color='w'))
        ax.add_patch(plt.Rectangle((size / 2 - line_width / 2, 0), line_width, size, color='k'))
        ax.add_patch(plt.Rectangle((0, size / 2 - line_width / 2), size, line_width, color='k'))
        ax.add_patch(plt.Circle((size / 2, size / 2), inner_diameter, color='w'))
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.show()

        self._pixel_array = data

        #save stimulus
        color_map = pyplot.get_cmap('gray')
        color_map.set_over(color="y")
        pyplot.imsave(fname = filename,
                    arr  = self._pixel_array,
                    cmap = color_map, format="png", vmin=0, vmax=1)
        #plt.savefig(filename = filename, cmap = color_map, format="png")

        self._background_colour = [0, 0, 0]

    @property
    def background_colour(self):
        """Getter for background_colour"""

        return self._background_colour


    @property
    def pixel_array(self):
        """Getter for pixel_array"""

        return self._pixel_array

if __name__ == "__main__":
    from .. import control, design, misc
    control.set_develop_mode(True)
    control.defaults.event_logging = 0
    garbor = GaborPatch(size=200, lambda_=10, theta=15,
                sigma=20, phase=0.25)
    exp = design.Experiment(background_colour=garbor.background_colour)
    control.initialize(exp)
    garbor.present()
    exp.clock.wait(1000)
