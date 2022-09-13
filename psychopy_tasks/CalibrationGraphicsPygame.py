#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR
# PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the distribution.
#
# Neither name of SR Research Ltd nor the name of contributors may be used
# to endorse or promote products derived from this software without
# specific prior written permission.
#
# Last updated on 3/18/2021

import pygame
from pygame.locals import *
from math import pi
import array
import pylink
import platform
import sys
import os


#allow to disable sound, or if we failed to initialize pygame.mixer or failed to load audio file
#continue experiment without sound.
DISABLE_AUDIO=False

class CalibrationGraphics(pylink.EyeLinkCustomDisplay):
    def __init__(self, tracker, win):
        global DISABLE_AUDIO
        pylink.EyeLinkCustomDisplay.__init__(self)

        self._disp = win  # screen to use for calibration
        self._tracker = tracker  # connection to the tracker

        self._version = '2021.3.16'
        self._last_updated = '3/16/2021'

        pygame.mouse.set_visible(False)  # hide mouse cursor

        self._bgColor = (128, 128, 128)  # target color (foreground)
        self._fgColor = (0, 0, 0)  # target color (background)
        self._targetSize = 32  # diameter of the target
        self._targetType = 'circle'  # could be 'circle' or 'picture'
        self._pictureTarget = None  # picture target

        self._target_beep = None
        self._done_beep = None
        self._error_beep = None
        
        if not DISABLE_AUDIO:
            try:
                self._target_beep = pygame.mixer.Sound("type.wav")
                self._done_beep = pygame.mixer.Sound("qbeep.wav")
                self._error_beep = pygame.mixer.Sound("error.wav")
            except Exception as e:
                print ('Failed to load audio: '+ str(e))
                #we failed to load audio, so disable it
                #if the experiment is run with sudo/root user in Ubuntu, then audio will
                #fail. The work around is either allow audio playback permission
                #for root user or,  run the experiment with non root user.
                DISABLE_AUDIO=True

        self._size = (384, 320)  # size of the camera image
        self._imagebuffer = array.array('I')  # buffer to store camera image
        self._resizedImg = None

        self.surf = pygame.display.get_surface()

        # image palette; its indices are used to reconstruct the camera image
        self._pal = []

        # we will use this for text messages
        self._fnt = pygame.font.SysFont('Arial', 26)
        self._w, self._h = self._disp.get_size()
        self._cam_region = pygame.Rect((0, 0), (0, 0))

        # cache the camera title
        self._title = ''

        # keep track of mouse states
        self.mouse_pos = (self._w/2, self._h/2)
        self.last_mouse_state = -1

    def __str__(self):
        """ overwrite __str__ to show some information about the
        CoreGraphicsPsychoPy library
        """

        return "Using the CalibrationGraphicsPygame library, " + \
            "version %s, " % self._version + \
            "last updated on %s" % self._last_updated

    def getForegroundColor(self):
        """ get the foreground color """

        return self._fgColor

    def getBackgroundColor(self):
        """ get the foreground color """

        return self._bgColor

    def setCalibrationColors(self, foreground_color, background_color):
        """ Set calibration background and foreground colors

        Parameters:
            foreground_color--foreground color for the calibration target
            background_color--calibration background.
            """
        self._fgColor = foreground_color
        self._bgColor = background_color

    def setTargetType(self, type):
        """ Set calibration target size in pixels

        Parameters:
            type: "circle" (default) or "picture"
        """

        self._targetType = type

    def setTargetSize(self, size):
        """ Set calibration target size in pixels"""

        self._targetSize = size

    def setPictureTarget(self, picture_target):
        """ set the movie file to use as the calibration target """

        self._pictureTarget = picture_target

    def setCalibrationSounds(self, target_beep, done_beep, error_beep):
        """ Provide three wav files as the warning beeps

        Parameters:
            target_beep -- sound to play when the target comes up
            done_beep -- calibration is done successfully
            error_beep -- calibration/drift-correction error.
        """

        # target beep
        if target_beep == '':
            self._target_beep = pygame.mixer.Sound("type.wav")
        elif target_beep == 'off':
            self._target_beep = None
        else:
            self._target_beep = pygame.mixer.Sound(target_beep)

        # done beep
        if done_beep == '':
            self._done_beep = pygame.mixer.Sound("qbeep.wav")
        elif done_beep == 'off':
            self._done_beep = None
        else:
            self._done_beep = pygame.mixer.Sound(done_beep)

        # error beep
        if error_beep == '':
            self._error_beep = pygame.mixer.Sound("error.wav")
        elif error_beep == 'off':
            self._error_beep = None
        else:
            self._error_beep = pygame.mixer.Sound(error_beep)

    def setup_cal_display(self):
        """ setup calibration/validation display""" 

        self.clear_cal_display()

    def exit_cal_display(self):
        """ exit calibration/validation display""" 

        self.clear_cal_display()

    def record_abort_hide(self):
        pass

    def clear_cal_display(self):
        self._disp.fill(self._bgColor)
        pygame.display.flip()
        self._disp.fill(self._bgColor)

    def erase_cal_target(self):
        self.clear_cal_display()

    def draw_cal_target(self, x, y):
        """  draw the calibration target, i.e., a bull's eye""" 

        if self._targetType == 'picture':
            if self._pictureTarget is None:
                print('ERROR: Provide a picture as the calibration target')
                pygame.quit()
                sys.exit()
            elif not os.path.exists(self._pictureTarget):
                print('ERROR: Picture %s not found' % self._pictureTarget)
                pygame.quit()
                sys.exit()
            else:
                cal_pic = pygame.image.load(self._pictureTarget)
                w, h = cal_pic.get_size()
                self._disp.blit(cal_pic, (x - int(w/2.0), y - int(h/2.0)))
        else:
            pygame.draw.circle(self._disp, self._fgColor, (x, y),
                               int(self._targetSize / 2.))
            pygame.draw.circle(self._disp, self._bgColor, (x, y),
                               int(self._targetSize / 4.))
        pygame.display.flip()

    def play_beep(self, beepid):
        """ play warning beeps if being requested""" 
        global DISABLE_AUDIO
        # if sound is disabled, don't play
        if DISABLE_AUDIO:
            pass
        else:
            if beepid in [pylink.DC_TARG_BEEP, pylink.CAL_TARG_BEEP]:
                if self._target_beep is not None:
                    self._target_beep.play()
                    pygame.time.wait(50)
            if beepid in [pylink.CAL_ERR_BEEP, pylink.DC_ERR_BEEP]:
                if self._error_beep is not None:
                    self._error_beep.play()
                    pygame.time.wait(300)
            if beepid in [pylink.CAL_GOOD_BEEP, pylink.DC_GOOD_BEEP]:
                if self._done_beep is not None:
                    self._done_beep.play()
                    pygame.time.wait(100)

    def getColorFromIndex(self, colorindex):
        """  color scheme for different elements """ 

        if colorindex == pylink.CR_HAIR_COLOR:
            return (255, 255, 255, 255)
        elif colorindex == pylink.PUPIL_HAIR_COLOR:
            return (255, 255, 255, 255)
        elif colorindex == pylink.PUPIL_BOX_COLOR:
            return (0, 255, 0, 255)
        elif colorindex == pylink.SEARCH_LIMIT_BOX_COLOR:
            return (255, 0, 0, 255)
        elif colorindex == pylink.MOUSE_CURSOR_COLOR:
            return (255, 0, 0, 255)
        else:
            return (0, 0, 0, 0)

    def draw_line(self, x1, y1, x2, y2, colorindex):
        """  draw lines""" 

        color = self.getColorFromIndex(colorindex)

        # get the camera image rect, then scale
        if self._size[0] > 192:
            imr = self._img.get_rect()
            x1 = int((float(x1) / 192) * imr.w)
            x2 = int((float(x2) / 192) * imr.w)
            y1 = int((float(y1) / 160) * imr.h)
            y2 = int((float(y2) / 160) * imr.h)
        # draw the line
        if True not in [x < 0 for x in [x1, x2, y1, y2]]:
            pygame.draw.line(self._img, color, (x1, y1), (x2, y2))

    def draw_lozenge(self, x, y, width, height, colorindex):
        """  draw the search limits with two lines and two arcs""" 

        color = self.getColorFromIndex(colorindex)

        if self._size[0] > 192:
            imr = self._img.get_rect()
            x = int((float(x) / 192) * imr.w)
            y = int((float(y) / 160) * imr.h)
            width = int((float(width) / 192) * imr.w)
            height = int((float(height) / 160) * imr.h)

        if width > height:
            rad = int(height / 2.)
            if rad == 0:
                return
            else:
                pygame.draw.line(self._img,
                                 color,
                                 (x + rad, y),
                                 (x + width - rad, y))
                pygame.draw.line(self._img,
                                 color,
                                 (x + rad, y + height),
                                 (x + width - rad, y + height))
                pygame.draw.arc(self._img,
                                color,
                                [x, y, rad*2, rad*2],
                                pi/2, pi*3/2, 1)
                pygame.draw.arc(self._img,
                                color,
                                [x+width-rad*2, y, rad*2, height],
                                pi*3/2, pi/2 + 2*pi, 1)
        else:
            rad = int(width / 2.)
            if rad == 0:
                return
            else:
                pygame.draw.line(self._img,
                                 color,
                                 (x, y + rad),
                                 (x, y + height - rad))
                pygame.draw.line(self._img,
                                 color,
                                 (x + width, y + rad),
                                 (x + width, y + height - rad))
                pygame.draw.arc(self._img,
                                color,
                                [x, y, rad*2, rad*2],
                                0, pi, 1)
                pygame.draw.arc(self._img,
                                color,
                                [x, y+height-rad*2, rad*2, rad*2],
                                pi, 2*pi, 1)

    def get_mouse_state(self):
        """  get mouse position and states""" 

        x, y = pygame.mouse.get_pos()
        state = pygame.mouse.get_pressed()
        x = x * self._size[0]/self._w/2.0
        y = y * self._size[1]/self._h/2.0

        return ((x, y), state[0])

    def get_input_key(self):
        """  handle key input and send it over to the tracker""" 

        ky = []
        for ev in pygame.event.get():

            # check keyboard events
            if ev.type == KEYDOWN:
                keycode = ev.key
                if keycode == K_F1:
                    keycode = pylink.F1_KEY
                elif keycode == K_F2:
                    keycode = pylink.F2_KEY
                elif keycode == K_F3:
                    keycode = pylink.F3_KEY
                elif keycode == K_F4:
                    keycode = pylink.F4_KEY
                elif keycode == K_F5:
                    keycode = pylink.F5_KEY
                elif keycode == K_F6:
                    keycode = pylink.F6_KEY
                elif keycode == K_F7:
                    keycode = pylink.F7_KEY
                elif keycode == K_F8:
                    keycode = pylink.F8_KEY
                elif keycode == K_F9:
                    keycode = pylink.F9_KEY
                elif keycode == K_F10:
                    keycode = pylink.F10_KEY
                elif keycode == K_PAGEUP:
                    keycode = pylink.PAGE_UP
                elif keycode == K_PAGEDOWN:
                    keycode = pylink.PAGE_DOWN
                elif keycode == K_UP:
                    keycode = pylink.CURS_UP
                elif keycode == K_DOWN:
                    keycode = pylink.CURS_DOWN
                elif keycode == K_LEFT:
                    keycode = pylink.CURS_LEFT
                elif keycode == K_RIGHT:
                    keycode = pylink.CURS_RIGHT
                elif keycode == K_BACKSPACE:
                    keycode = ord('\b')
                elif keycode == K_RETURN:
                    keycode = pylink.ENTER_KEY
                    # probe the tracker to see if it's "simulating gaze
                    # with mouse". if so, show a warning instead of a blank
                    # screen to experimenter do so, only when the tracker
                    # is in Camera Setup screen
                    if self._tracker.getCurrentMode() == pylink.IN_SETUP_MODE:
                        self._tracker.readRequest('aux_mouse_simulation')
                        pylink.pumpDelay(50)
                        if self._tracker.readReply() == '1':
                            # draw a rectangle to mark the camera image
                            rec_x = int((self._w - 192*2) / 2.0)
                            rec_y = int((self._h - 160*2) / 2.0)
                            rct = pygame.Rect((rec_x, rec_y, 192*2, 160*2))
                            pygame.draw.rect(self._disp, self._fgColor, rct, 2)
                            # show some message
                            msg = 'Simulating gaze with the mouse'
                            msg_w, msg_h = self._fnt.size(msg)
                            t_surf = self._fnt.render(msg, True, self._fgColor)

                            txt_x = int((self._w - msg_w)/2.0)
                            txt_y = int((self._h - msg_h)/2.0)
                            self._disp.blit(t_surf, (txt_x, txt_y))
                            pygame.display.flip()
                elif keycode == K_SPACE:
                    keycode = ord(' ')
                elif keycode == K_ESCAPE:
                    keycode = pylink.ESC_KEY
                elif keycode == K_TAB:
                    keycode = ord('\t')
                elif(keycode == pylink.JUNK_KEY):
                    keycode = 0

                ky.append(pylink.KeyInput(keycode, ev.mod))

        return ky

    def exit_image_display(self):
        """  exit the camera image display""" 

        self.clear_cal_display()

    def alert_printf(self, msg):
        print(msg)

    def setup_image_display(self, width, height):
        """  set up the camera image display

        return 1 to request high-resolution camera image""" 

        self._size = (width, height)
        self.clear_cal_display()
        self.last_mouse_state = -1

        return 1

    def image_title(self, text):
        """  show the camera image title

        target distance, and pupil/CR thresholds below the image. To prevent
        drawing glitches, we cache the image title and draw it with the camera
        image in the draw_image_line function instead""" 

        self._title = text

    def draw_image_line(self, width, line, totlines, buff):
        """  draw the camera image""" 

        for i in range(width):
            try:
                self._imagebuffer.append(self._pal[buff[i]])
            except:
                pass

        if line == totlines:
            try:
                # construct the camera image from the buffer
                try:
                    tmp_buffer = self._imagebuffer.tobytes()
                except:
                    tmp_buffer = self._imagebuffer.tostring()
                    
                cam = pygame.image.frombuffer(tmp_buffer,
                                              (width, totlines), 'RGBX')
                self._img = cam
                self.draw_cross_hair()

                # prepare the camera image
                img_w, img_h = (width*2, totlines*2)
                self._resizedImg = pygame.transform.scale(cam, (img_w, img_h))
                cam_img_pos = ((self._w/2-img_w/2),
                               (self._h/2-img_h/2))

                # prepare the camera image caption
                txt_w, txt_h = self._fnt.size(self._title)
                txt_surf = self._fnt.render(self._title, True, self._fgColor)
                txt_pos = (int(self._w/2 - txt_w/2),
                           int(self._h/2 + img_h/2 + txt_h/2))

                # draw the camera image and the caption
                surf = pygame.display.get_surface()
                surf.fill(self._bgColor)
                surf.blit(self._resizedImg, cam_img_pos)
                surf.blit(txt_surf, txt_pos)
                pygame.display.flip()
            except:
                pass

            self._imagebuffer = array.array('I')

    def set_image_palette(self, r, g, b):
        """  get the color palette for the camera image""" 

        self._imagebuffer = array.array('I')

        sz = len(r)
        i = 0
        self._pal = []
        while i < sz:
            rf = int(b[i])
            gf = int(g[i])
            bf = int(r[i])
            self._pal.append((rf << 16) | (gf << 8) | (bf))
            i = i + 1

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

    # initialize Pygame
    pygame.init()

    # get the screen resolution natively supported by the monitor
    disp = pylink.getDisplayInformation()
    scn_w = disp.width
    scn_h = disp.height

    # connect to the tracker
    el_tracker = pylink.EyeLink("100.1.1.1")

    # open an EDF data file on the Host PC
    el_tracker.openDataFile('test.edf')

    # open a Pygame window
    win = pygame.display.set_mode((scn_w, scn_h), FULLSCREEN | DOUBLEBUF)

    # send over a command to let the tracker know the correct screen resolution
    scn_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_w - 1, scn_h - 1)
    el_tracker.sendCommand(scn_coords)

    # Instantiate a graphics environment (genv) for calibration
    genv = CalibrationGraphics(el_tracker, win)

    # Set background and foreground colors for calibration
    foreground_color = (0, 0, 0)
    background_color = (128, 128, 128)
    genv.setCalibrationColors(foreground_color, background_color)

    # The calibration target could be a "circle" (default) or a "picture",
    genv.setTargetType('circle')
    # Configure the size of the calibration target (in pixels)
    genv.setTargetSize(24)

    # Beeps to play during calibration, validation, and drift correction
    # parameters: target, good, error
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    genv.setCalibrationSounds('', '', '')

    # Request Pylink to use the graphics environment (genv) we customized above
    pylink.openGraphicsEx(genv)

    # calibrate the tracker
    el_tracker.doTrackerSetup()

    # close the data file
    el_tracker.closeDataFile()

    # disconnect from the tracker
    el_tracker.close()

    # quit pygame
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
