#
# Copyright (C) 2018-2021 SR Research Ltd.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at
# your option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Last updated on 5/13/2021

from __future__ import division
from __future__ import print_function

import os
import platform
import array
import string
import pylink
import numpy
import psychopy
from psychopy import visual, event, core, logging, prefs, monitors
from psychopy.tools.coordinatetools import pol2cart
from math import sin, cos, pi
from PIL import Image, ImageDraw
from psychopy.sound import Sound


#allow to disable sound, or if we failed to initialize pygame.mixer or failed to load audio file
#continue experiment without sound.
DISABLE_AUDIO=True


# Show only critical log message in the console
logging.console.setLevel(logging.CRITICAL)


class EyeLinkCoreGraphicsPsychoPy(pylink.EyeLinkCustomDisplay):
    def __init__(self, tracker, win):

        """ Constructor for Custom EyeLinkCoreGraphics

        tracker: an EyeLink instance (connection)
        win: the Psychopy display we use for stimulus presentation""" 
        global DISABLE_AUDIO
        pylink.EyeLinkCustomDisplay.__init__(self)

        self._version = '2021.3.31'
        self._last_updated = '3/31/2021'

        # Calibration background color and target color
        self._backgroundColor = win.color
        self._foregroundColor = 'black'

        # Get the major version # for PsychoPy, from version 2021.1.4
        # The major version is the year of release
        self._psychopyVer = int(psychopy.__version__.split('.')[0])

        self._display = win
        self._display.mouseVisible = False  # set mouse cursor invisible

        self._display.autoLog = False  # disable windows msg logging
        self._w, self._h = win.size

        # Forcing the screen units to 'pix'
        self._units = win.units
        if self._units != 'pix':
            self._display.setUnits('pix')

        # Camera image set up
        self._imagebuffer = array.array('I')
        self._pal = None  # color pallete to use for camera image drawing
        self._size = (384, 320)

        # Initial setup for the mouse
        self._mouse = event.Mouse(False)
        self.last_mouse_state = -1

        # Image title & calibration instructions
        self._msgHeight = self._size[1]/16.0
        __title_pos__ = (0, -self._size[1]/2 - self._msgHeight)
        self._title = visual.TextStim(self._display, '',
                                      height=self._msgHeight,
                                      color=[1, 1, 1],
                                      pos=__title_pos__,
                                      wrapWidth=self._w,
                                      units='pix')

        calib_instruction = 'Enter: Show/Hide camera image\n' + \
                            'Left/Right: Switch camera view\n' + \
                            'C: Calibration\n' + \
                            'V: Validation\n' + \
                            'O: Start Recording\n' + \
                            '+=/-: CR threshold\n' + \
                            'Up/Down: Pupil threshold\n' + \
                            'Alt+arrows: Search limit'
        __calibInst_pos__ = (20 - self._w/2, self._h/2 - 20)
        self._calibInst = visual.TextStim(self._display,
                                          height=self._msgHeight,
                                          color=[1, 1, 1],
                                          pos=__calibInst_pos__,
                                          units='pix',
                                          text=calib_instruction)

        # Show some instruction to let the experimenter know that the tracker
        # is running in mouse simulation mode
        __mouse_sim_msg__ = 'Simulating gaze using the mouse\n\n' + \
                            'NO CAMERA IMAGE IS AVAILABLE'
        self._msgMouseSim = visual.TextStim(self._display,
                                            height=self._msgHeight,
                                            color=[1, 1, 1],
                                            units='pix',
                                            text=__mouse_sim_msg__)
        # Use an empty square to mark the camera image, when the tracker is
        # running in mouse simulation mode
        self._camImgRect = visual.Rect(self._display,
                                       width=self._size[0],
                                       height=self._size[1],
                                       lineColor=[1, 1, 1],
                                       units='pix')

        # alignHoriz and alignVert were deprecated from version 2021
        # (also know as PsychoPy 3.3),
        # Use anchorHoriz & anchorVer for text formatting instead
        if self._psychopyVer > 3:
            self._calibInst.alignText = 'left'
            self._calibInst.anchorHoriz = 'left'
            self._calibInst.anchorVert = 'top'
        else:
            self._calibInst.alignHoriz = 'left'
            self._calibInst.alignVert = 'top'

        # Configure the calibration target
        self._targetSize = self._w/64.  # default size 1/64 of screen width
        # Target could be 'circle', 'picture', 'spiral', 'movie'
        self._calTarget = 'circle'
        # A switch to turn on/off the animated target
        self._animatedTarget = False
        self._movieTarget = None
        self._pictureTarget = None

        # Configure calibration sounds (beeps), use ".wav" files
        if not DISABLE_AUDIO:
            try:
                self._target_beep = Sound('type.wav', stereo=True)
                self._error_beep = Sound('error.wav', stereo=True)
                self._done_beep = Sound('qbeep.wav', stereo=True)
            except Exception as e:
                print ('Failed to load audio: '+ str(e))
                #we failed to load audio, so disable it
                #if the experiment is run with sudo/root user in Ubuntu, then audio will
                #fail. The work around is either allow audio playback permission
                #for root user or,  run the experiment with non root user.
                DISABLE_AUDIO=True

        # A reference to the tracker connection
        self._tracker = tracker

        # The tracker is running in mouse simulation mode?
        self._mouse_simulation = False
        self.imgResize = None

    def __str__(self):
        """ Overwrite __str__ to show some information about the
        CoreGraphicsPsychoPy library
        """

        return "Using the EyeLinkCoreGraphicsPsychoPy library, " + \
            "version %s, " % self._version + \
            "last updated on %s" % self._last_updated

    def fixMacRetinaDisplay(self):
        """ Fix macOS retina display issue """

        # Resolution fix for Mac retina displays
        if 'Darwin' in platform.system():
            self._w = int(self._w/ 2.0)
            self._h = int(self._h / 2.0)
            self._calibInst.pos =  (20 - self._w/2, self._h/2 - 20)

    def getForegroundColor(self):
        """ Get the foreground color """

        return self._foregroundColor

    def getBackgroundColor(self):
        """ Get the foreground color """

        return self._backgroundColor

    def setCalibrationColors(self, foreground_color, background_color):
        """ Set calibration background and foreground colors

        Parameters:
            foreground_color--foreground color for the calibration target
            background_color--calibration background.
        """

        self._foregroundColor = foreground_color
        self._backgroundColor = background_color

        # Update the color of the visual elements
        self._title.color = foreground_color
        self._calibInst.color = foreground_color
        self._display.color = background_color
        self._msgMouseSim.color = foreground_color
        self._camImgRect.lineColor = foreground_color

    def setTargetSize(self, size):
        """ Set calibration target size in pixels"""

        self._targetSize = size

    def setTargetType(self, type):
        """ Set calibration target size in pixels

        Parameters:
            type: "circle" (default), "picture", "movie", "spiral"
        """

        self._calTarget = type

    def setMoiveTarget(self, movie_target):
        """ Set the movie file to use as the calibration target """

        self._movieTarget = movie_target
        
    def setPictureTarget(self, picture_target):
        """ Set the movie file to use as the calibration target """

        self._pictureTarget = picture_target

    def setCalibrationSounds(self, target_beep, done_beep, error_beep):
        """ Provide three wav files as the warning beeps

        Parameters:
            target_beep -- sound to play when the target comes up
            done_beep -- calibration is done successfully
            error_beep -- calibration/drift-correction error.
        """

        # Target beep
        if target_beep == '':
            pass
        elif target_beep == 'off':
            self._target_beep = None
        else:
            self._target_beep.setSound(target_beep)

        # Done beep
        if done_beep == '':
            pass
        elif done_beep == 'off':
            self._done_beep = None
        else:
            self._done_beep.setSound(done_beep)

        # Error beep
        if error_beep == '':
            pass
        elif error_beep == 'off':
            self._error_beep = None
        else:
            self._error_beep.setSound(error_beep)

    def update_cal_target(self):
        """ Make sure target stimuli is already memory when
            being used by draw_cal_target """ 

        if self._calTarget == 'picture':
            if self._pictureTarget is None:
                print('ERROR: Provide a picture as the calibration target')
                core.quit()
                sys.exit()
            else:
                if os.path.exists(self._pictureTarget):
                    self._calibTar = visual.ImageStim(self._display,
                                                      self._pictureTarget)
                else:
                    print("ERROR: Picture %s not found" % self._pictureTarget)
                    self._display.close()
                    core.quit()
        elif self._calTarget == 'spiral':
            thetas = numpy.arange(0, 1440, 10)
            N = len(thetas)
            radii = numpy.linspace(0, 1.0, N)*self._targetSize
            x, y = pol2cart(theta=thetas, radius=radii)
            xys = numpy.array([x, y]).transpose()
            self._calibTar = visual.ElementArrayStim(self._display,
                                                     nElements=N,
                                                     sizes=self._targetSize,
                                                     sfs=3.0,
                                                     xys=xys,
                                                     oris=-thetas)

        elif self._calTarget == 'movie':
            if self._movieTarget is None:
                print('ERROR: Provide a movie clip as the calibration target')
                core.quit()
            else:
                if os.path.exists(self._movieTarget):
                    self._calibTar = visual.MovieStim3(self._display,
                                                       self._movieTarget,
                                                       noAudio=False,
                                                       loop=True)
                else:
                    print("ERROR: Movie %s not found" % self._movieTarget)
                    self._display.close()
                    core.quit()
        else:  # Use the default target 'circle'
            self._tarOuter = visual.GratingStim(self._display,
                                                tex='none',
                                                mask='circle',
                                                size=self._targetSize,
                                                color=self._foregroundColor,
                                                units='pix')
            self._tarInner = visual.GratingStim(self._display,
                                                tex='none',
                                                mask='circle',
                                                size=self._targetSize/2,
                                                color=self._backgroundColor,
                                                units='pix')

    def setup_cal_display(self):
        """ Set up the calibration display before entering
        the calibration/validation routine""" 

        self._display.clearBuffer()

        self._calibInst.autoDraw = True
        self._animatedTarget = False
        self.update_cal_target()
        
    def clear_cal_display(self):
        """ Clear the calibration display""" 

        self._calibInst.autoDraw = False
        self._title.autoDraw = False
        self._msgMouseSim.autoDraw = False
        self._camImgRect.autoDraw = False

        self._display.color = self._backgroundColor
        self._display.flip()
        self._display.color = self._backgroundColor

    def exit_cal_display(self):
        """ Exit the calibration/validation routine, set the screen
        units to the original one used by the user""" 

        self._display.setUnits(self._units)
        self._animatedTarget = False
        self.clear_cal_display()

    def record_abort_hide(self):
        """ This function is called if aborted""" 

        pass

    def erase_cal_target(self):
        """ Erase the calibration/validation & drift-check target""" 

        try:
            self._calibTar.pause()
        except:
            pass
        self.clear_cal_display()
        self._animatedTarget = False
        self._display.flip()

    def draw_cal_target(self, x, y):
        """ Draw the calibration/validation & drift-check  target""" 

        self._calibInst.autoDraw = False

        self.clear_cal_display()
        xVis = (x - self._w/2.0)
        yVis = (self._h/2.0 - y)

        # Update the target position
        if self._calTarget == 'circle':
            self._tarOuter.pos = (xVis, yVis)
            self._tarInner.pos = (xVis, yVis)
        elif self._calTarget == 'spiral':
            self._calibTar.fieldPos = (xVis, yVis)
        else:
            if self._calibTar is not None:
                self._calibTar.pos = (xVis, yVis)

        # Handle the drawing
        if self._calTarget in ['spiral', 'movie']:
            # Hand over drawing to get_input_key()
            self._animatedTarget = True
            if self._calTarget == 'movie':
                if self._calibTar is not None:
                    self._calibTar.play()
        elif self._calTarget == 'picture':
            self._calibTar.draw()
            self._display.flip()
        else:
            self._tarOuter.draw()
            self._tarInner.draw()
            self._display.flip()
        
    def play_beep(self, beepid):
        """ Play a sound during calibration/drift correct.""" 

        global DISABLE_AUDIO
        # if sound is disabled, don't play
        if DISABLE_AUDIO:
            pass
        else:
            if self._calTarget == 'movie':
                pass
            else:
                if beepid in [pylink.CAL_TARG_BEEP, pylink.DC_TARG_BEEP]:
                    if self._target_beep is not None:
                        self._target_beep.play()
                        core.wait(0.5)
                elif beepid in [pylink.CAL_ERR_BEEP, pylink.DC_ERR_BEEP]:
                    if self._error_beep is not None:
                        self._error_beep.play()
                        core.wait(1.2)
                elif beepid in [pylink.CAL_GOOD_BEEP, pylink.DC_GOOD_BEEP]:
                    if self._done_beep is not None:
                        self._done_beep.play()
                        core.wait(0.5)
                else:
                    pass

    def getColorFromIndex(self, colorindex):
        """ Return psychopy colors for elements in the camera image""" 

        if colorindex == pylink.CR_HAIR_COLOR:
            return (255, 255, 255)
        elif colorindex == pylink.PUPIL_HAIR_COLOR:
            return (255, 255, 255)
        elif colorindex == pylink.PUPIL_BOX_COLOR:
            return (0, 255, 0)
        elif colorindex == pylink.SEARCH_LIMIT_BOX_COLOR:
            return (255, 0, 0)
        elif colorindex == pylink.MOUSE_CURSOR_COLOR:
            return (255, 0, 0)
        else:
            return (128, 128, 128)

    def draw_line(self, x1, y1, x2, y2, colorindex):
        """ Draw a line. This is used for drawing crosshairs/squares""" 

        color = self.getColorFromIndex(colorindex)

        if self._size[0] > 192:
            w, h = self._img.im.size
            x1 = int((float(x1) / 192) * w)
            x2 = int((float(x2) / 192) * w)
            y1 = int((float(y1) / 160) * h)
            y2 = int((float(y2) / 160) * h)

        # draw the line
        if not any([x < 0 for x in [x1, x2, y1, y2]]):
            self._img.line([(x1, y1), (x2, y2)], color)

    def draw_lozenge(self, x, y, width, height, colorindex):
        """ Draw a lozenge to show the defined search limits
        (x,y) is top-left corner of the bounding box
        """ 

        color = self.getColorFromIndex(colorindex)

        if self._size[0] > 192:
            w, h = self._img.im.size
            x = int((float(x) / 192) * w)
            y = int((float(y) / 160) * h)
            width = int((float(width) / 192) * w)
            height = int((float(height) / 160) * h)

        if width > height:
            rad = int(height / 2.)
            if rad == 0:
                return
            else:
                self._img.line([(x + rad, y), (x + width - rad, y)], color, 1)
                self._img.line([(x + rad, y + height),
                                (x + width - rad, y + height)], color, 1)
                self._img.arc([x, y, x + rad*2, y + rad*2], 90, 270, color, 1)
                self._img.arc([x + width - rad*2, y, x + width, y + height],
                              270, 90, color, 1)
        else:
            rad = int(width / 2.)
            if rad == 0:
                return
            else:
                self._img.line([(x, y + rad), (x, y + height - rad)], color, 1)
                self._img.line([(x + width, y + rad),
                                (x + width, y + height - rad)], color, 1)
                self._img.arc([x, y, x + rad*2, y + rad*2], 180, 360, color, 1)
                self._img.arc([x, y + height-rad*2, x + rad*2, y + height],
                              0, 180, color, 1)

    def get_mouse_state(self):
        """ Get the current mouse position and status""" 

        w, h = self._display.size
        X, Y = self._mouse.getPos()
        mX = (X + w/2.0)/w*self._size[0]/2.0
        mY = (h/2.0 - Y)/h*self._size[1]/2.0

        state = self._mouse.getPressed()[0]

        return ((mX, mY), state)

    def get_input_key(self):
        """ This function will be constantly pools, update the stimuli
        here is you need dynamic calibration target """ 

        # This function is constantly checked by the API,
        # so we could update the gabor here
        if self._animatedTarget:
            if self._calTarget == 'spiral':
                self._calibTar.phases -= 0.02
            self._calibTar.draw()
            self._display.flip()
        
        ky = []
        for keycode, modifier in event.getKeys(modifiers=True):
            self._display.mouseVisible = False  # set mouse cursor invisible
            k = pylink.JUNK_KEY
            if keycode == 'f1':
                k = pylink.F1_KEY
            elif keycode == 'f2':
                k = pylink.F2_KEY
            elif keycode == 'f3':
                k = pylink.F3_KEY
            elif keycode == 'f4':
                k = pylink.F4_KEY
            elif keycode == 'f5':
                k = pylink.F5_KEY
            elif keycode == 'f6':
                k = pylink.F6_KEY
            elif keycode == 'f7':
                k = pylink.F7_KEY
            elif keycode == 'f8':
                k = pylink.F8_KEY
            elif keycode == 'f9':
                k = pylink.F9_KEY
            elif keycode == 'f10':
                k = pylink.F10_KEY
            elif keycode == 'pageup':
                k = pylink.PAGE_UP
            elif keycode == 'pagedown':
                k = pylink.PAGE_DOWN
            elif keycode == 'up':
                k = pylink.CURS_UP
            elif keycode == 'down':
                k = pylink.CURS_DOWN
            elif keycode == 'left':
                k = pylink.CURS_LEFT
            elif keycode == 'right':
                k = pylink.CURS_RIGHT
            elif keycode == 'backspace':
                k = ord('\b')
            elif keycode == 'return':
                k = pylink.ENTER_KEY
                # Probe the tracker to see if it's "simulating gaze with mouse"
                # If so, show a warning to experimenter
                if self._tracker.getCurrentMode() == pylink.IN_SETUP_MODE:
                    self._tracker.readRequest('aux_mouse_simulation')
                    pylink.pumpDelay(50)
                    if self._tracker.readReply() == '1':
                        self._msgMouseSim.autoDraw = True
                        self._camImgRect.autoDraw = True
                        self._calibInst.autoDraw = True
                        self._display.flip()
            elif keycode == 'space':
                k = ord(' ')
            elif keycode == 'escape':
                k = 27
            elif keycode == 'tab':
                k = ord('\t')
            elif keycode in string.ascii_letters:
                k = ord(keycode)
            elif k == pylink.JUNK_KEY:
                k = 0

            # Plus/equal & minux signs for CR adjustment
            if keycode in ['num_add', 'equal']:
                k = ord('+')
            if keycode in ['num_subtract', 'minus']:
                k = ord('-')

            # Handles key modifier, we can send Ctrl-C, Alt-F4
            # to break out trials, or terminate tasks
            if modifier['alt'] is True:
                mod = 256
            elif modifier['ctrl'] is True:
                mod = 64
            elif modifier['shift'] is True:
                mod = 1
            else:
                mod = 0

            ky.append(pylink.KeyInput(k, mod))

        return ky

    def exit_image_display(self):
        """ Clear the camera image""" 

        self._calibInst.autoDraw = True
        self._title.autoDraw = False
        self._msgMouseSim.autoDraw = False
        self._camImgRect.autoDraw = False
        self._display.flip()

    def alert_printf(self, msg):
        """ Print error messages.""" 

        print("Error: " + msg)

    def setup_image_display(self, width, height):
        """ Set up the camera image, for newer APIs,
        the size is 384 x 320 pixels""" 

        self.last_mouse_state = -1
        self._calibInst.autoDraw = True
        self._title.autoDraw = True
        self._msgMouseSim.autoDraw = False
        self._camImgRect.autoDraw = False

        self._size = (width, height)

        return 1

    def image_title(self, text):
        """ Draw title text below the camera image""" 

        if self.imgResize is not None:
            im_w, im_h = self.imgResize.size
            self._title.pos = (0, - im_h/2.0 - self._msgHeight)
        else:
            self._title.pos = (0, -self._size[1]/2 - self._msgHeight)
        self._title.text = text

    def draw_image_line(self, width, line, totlines, buff):
        """ Display image pixel by pixel, line by line""" 

        i = 0
        for i in range(width):
            try:
                self._imagebuffer.append(self._pal[buff[i]])
            except:
                pass

        if line == totlines:
            bufferv = self._imagebuffer.tostring()
            img = Image.frombytes("RGBX", (width, totlines), bufferv)
            self._img = ImageDraw.Draw(img)
            self.draw_cross_hair()
            self.imgResize = img.resize((width*2, totlines*2))
            imgResizeVisual = visual.ImageStim(self._display,
                                               image=self.imgResize,
                                               units='pix')
            imgResizeVisual.draw()
            # Change the position of the camera title
            self._title.pos = (0, - totlines*2/2.0 - self._msgHeight)
            self._display.flip()
            self._imagebuffer = array.array('I')

    def set_image_palette(self, r, g, b):
        """ Given a set of RGB colors, create a list of 24bit numbers
        representing the pallet.

        i.e., RGB of (1,64,127) would be saved as 82047,
        or the number 00000001 01000000 011111111""" 

        self._imagebuffer = array.array('I')

        sz = len(r)
        i = 0
        self._pal = []
        while i < sz:
            rf = int(b[i])
            gf = int(g[i])
            bf = int(r[i])
            self._pal.append((rf << 16) | (gf << 8) | (bf))
            i = i+1


# A short testing script showing the basic usage of this library
# We first instantiate a connection to the tracker (el_tracker), then we open
# a Pygame window (win). We then pass the tracker connection and the Pygame
# window to the graphics environment constructor (CalibrationGraphics).
# The graphics environment, once instantiated, can be configured to customize
# the calibration foreground and background color, the calibration target
# type, the calibration target size, and the beeps we would like to
# play during calibration and validation.
#
# IMPORTANT: Once the graphics environment is properly configured, call the
# pylink.openGraphicsEx() function to request Pylink to use the custom graphics
# environment for calibration instead.

def main():
    """ A short script showing how to use this library.

    We connect to the tracker, open a Pygame window, and then configure the
    graphics environment for calibration. Then, perform a calibration and
    disconnect from the tracker.

    The doTrackerSetup() command will bring up a gray calibration screen.
    When the gray screen comes up, press Enter to show the camera image,
    press C to calibrate, V to validate, and O to quit calibration"""

    # Set the screen resolution
    scn_w, scn_h = (1440, 900)

    # Connect to the tracker
    el_tracker = pylink.EyeLink("100.1.1.1")

    # Open an EDF data file on the Host PC
    el_tracker.openDataFile('test.edf')

    # Open a window, be sure to specify the monitor resolution
    mon = monitors.Monitor('myMonitor', width=53.0, distance=70.0)
    mon.setSizePix((scn_w, scn_h))
    win = visual.Window((scn_w, scn_h),
                        fullscr=True,
                        monitor=mon,
                        winType='pyglet',
                        units='pix')

    # Send over a command to let the tracker know the correct screen resolution
    scn_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_w - 1, scn_h - 1)
    el_tracker.sendCommand(scn_coords)

    # Instantiate a graphics environment (genv) for calibration
    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)

    # Set background and foreground colors for calibration
    foreground_color = (-1, -1, -1)
    background_color = win.color
    genv.setCalibrationColors(foreground_color, background_color)

    # The target could be a "circle" (default), a "picture", a "movie" clip,
    # or a rotating "spiral".
    genv.setTargetType('circle')
    # Configure the size of the calibration target (in pixels)
    genv.setTargetSize(24)

    # Beeps to play during calibration, validation and drift correction
    # parameters: target, good, error
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    genv.setCalibrationSounds('', '', '')

    # Request Pylink to use the graphics environment (genv) we customized above
    pylink.openGraphicsEx(genv)

    # Calibrate the tracker
    el_tracker.doTrackerSetup()

    # Close the data file
    el_tracker.closeDataFile()

    # Disconnect from the tracker
    el_tracker.close()

    # Quit pygame
    core.quit()
    sys.exit()

if __name__ == '__main__':
    main()
