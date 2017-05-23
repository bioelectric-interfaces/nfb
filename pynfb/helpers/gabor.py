#!/usr/bin/env python

"""
A Gabor patch stimulus.

This module contains a class implementing a Gabor patch stimulus.

"""
from __future__ import absolute_import, print_function, division
from builtins import *

__author__ = 'Florian Krause <florian@expyriment.org>, \
Oliver Lindemann <oliver@expyriment.org>'
__version__ = '0.9.0'
__revision__ = 'c4963ac'
__date__ = 'Thu Mar 9 13:48:59 2017 +0100'

import os
import tempfile
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


class GaborPatch(Picture):
    """A class implementing a Gabor Patch."""

    def __init__(self, size=None, position=None, lambda_=None, theta=None,
                sigma=None, phase=None, trim=None, contrast=1.):
        """Create a Gabor Patch.

        Parameters
        ----------
        size : (int, int), optional
            size (x, y) of the mask
        position  : (int, int), optional
            position of the mask stimulus
        lambda_ : int, optional
            Spatial frequency (pixel per cycle)
        theta : int or float, optional
            Grating orientation in degrees
        sigma : int or float, optional
            gaussian standard deviation (in pixels)
        phase : float
            0 to 1 inclusive

        Notes
        -----
        The background colour of the stimulus depends of the parameters of
        the Gabor patch and can be determined (e.g. for plotting) with the
        property `GaborPatch.background_colour`.

        """

        # Parts of the code has be ported from http://www.icn.ucl.ac.uk/courses/MATLAB-Tutorials/Elliot_Freeman/html/gabor_tutorial.html

        if not isinstance(np, ModuleType):
            message = """GaborPatch can not be initialized.
The Python package 'Numpy' is not installed."""
            raise ImportError(message)

        if not isinstance(pyplot, ModuleType):
            message = """GaborPatch can not be initialized.
The Python package 'Matplotlib' is not installed."""
            raise ImportError(message)

        if size is None:
            size = defaults.gaborpatch_size
        if position is None:
            position = defaults.gaborpatch_position
        if lambda_ is None:
            lambda_ = defaults.gaborpatch_lambda_
        if theta is None:
            theta = defaults.gaborpatch_theta
        if sigma is None:
            sigma = defaults.gaborpatch_sigma
        if phase is None:
            phase = defaults.gaborpatch_phase


        fid, filename = tempfile.mkstemp(
                    dir=stimuli.defaults.tempdir,
                    suffix=".png")
        os.close(fid)
        Picture.__init__(self, filename, position)

        # make linear ramp
        X0 = (np.linspace(1, size, size) / size) - .5
        # Set wavelength and phase
        freq = size / float(lambda_)
        phaseRad = phase * 2 * np.pi
        # Make 2D grating
        Xm, Ym = np.meshgrid(X0, X0)
        # Change orientation by adding Xm and Ym together in different proportions
        thetaRad = (theta / 360.) * 2 * np.pi
        Xt = Xm * np.cos(thetaRad)
        Yt = Ym * np.sin(thetaRad)
        grating = (np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad) + 1) / 2
        # 2D Gaussian distribution
        gauss = np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(size)) ** 2))
        print(np.max(gauss), np.max(grating),np.min(grating))
        # Trim
        #gauss[gauss < trim] = 0

        self._pixel_array = grating * gauss * contrast

        #save stimulus
        color_map = pyplot.get_cmap('gray')
        color_map.set_over(color="y")
        #print(gauss)
        #pyplot.imshow(gauss,        cmap = color_map)
        #pyplot.show()
        #pyplot.imshow(grating, cmap=color_map)
        #pyplot.show()
        pyplot.imsave(fname = filename,
                    arr  = self._pixel_array,
                    cmap = color_map, format="png", vmin=0, vmax=1)

        # determine background color
        norm = pyplot.Normalize(vmin = np.min(self._pixel_array),
                                vmax = np.max(self._pixel_array))
        bgc = color_map(norm(0))
        self._background_colour = [int(x*255) for x in bgc[:3]]

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
