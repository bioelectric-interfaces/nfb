import os
import sys
import logging
import time
import pylink
import platform
import pygame
from string import ascii_letters, digits

from PyQt5 import QtCore, QtGui, QtWidgets

from pynfb.experiment import Experiment
from pynfb.serializers.xml_ import xml_file_to_params
from pynfb.settings_widget.general import GeneralSettingsWidget
from pynfb.settings_widget.inlet import InletSettingsWidget
from pynfb.settings_widget.protocol_sequence import ProtocolSequenceSettingsWidget
from pynfb.settings_widget.protocols import ProtocolsSettingsWidget, FileSelectorLine
from pynfb.settings_widget.signals import SignalsSettingsWidget
from pynfb.settings_widget.composite_signals import CompositeSignalsSettingsWidget
from pynfb.settings_widget.protocols_group import ProtocolGroupsSettingsWidget

from psychopy_tasks.CalibrationGraphicsPygame import CalibrationGraphics

static_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static')


class SettingsWidget(QtWidgets.QWidget):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        v_layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        self.params = xml_file_to_params()
        self.general_settings = GeneralSettingsWidget(parent=self)
        v_layout.addWidget(self.general_settings)
        v_layout.addLayout(layout)
        self.protocols_list = ProtocolsSettingsWidget(parent=self)
        self.signals_list = SignalsSettingsWidget(parent=self)
        self.composite_signals_list = CompositeSignalsSettingsWidget(parent=self)
        self.protocol_groups_list = ProtocolGroupsSettingsWidget(parent=self)
        self.protocols_sequence_list = ProtocolSequenceSettingsWidget(parent=self)
        # layout.addWidget(self.general_settings)
        layout.addWidget(self.signals_list)
        layout.addWidget(self.composite_signals_list)
        layout.addWidget(self.protocols_list)
        layout.addWidget(self.protocol_groups_list)
        layout.addWidget(self.protocols_sequence_list)
        eye_calibrate_button = QtWidgets.QPushButton('Calibrate eye-link')
        eye_calibrate_button.clicked.connect(self.calibrateEyeLink)
        start_button = QtWidgets.QPushButton('Start')
        start_button.setIcon(QtGui.QIcon(static_path + '/imag/power-button.png'))
        start_button.setMinimumHeight(50)
        start_button.setMinimumWidth(300)
        start_button.clicked.connect(self.onClicked)
        name_layout = QtWidgets.QHBoxLayout()
        v_layout.addWidget(eye_calibrate_button, alignment=QtCore.Qt.AlignCenter)
        v_layout.addWidget(start_button, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(v_layout)
        self.setMinimumSize(self.layout().minimumSize())

    def sizeHint(self):
        return self.minimumSize()

    def reset_parameters(self):
        self.signals_list.reset_items()
        self.composite_signals_list.reset_items()
        self.protocols_list.reset_items()
        self.protocols_sequence_list.reset_items()
        self.protocol_groups_list.reset_items()
        self.general_settings.reset()
        # self.params['sExperimentName'] = self.experiment_name.text()

    def onClicked(self):
        self.experiment = Experiment(self.app, self.params)

    def calibrateEyeLink(self):
        # =============DO THE EYE TRACKER SETUP AND CALIBRATION
        # ...............................................................................
        pygame.init()
        # Set this variable to True to run the script in "Dummy Mode"
        dummy_mode = False

        # Workaround for pygame 2.0 shows black screen when running in full
        # screen mode in linux
        full_screen = True

        if 'Linux' in platform.platform():
            if int(pygame.version.ver[0]) > 1:
                full_screen = False

        # get the screen resolution natively supported by the monitor
        scn_width, scn_height = 0, 0

        # Store the parameters of all trials in a list, [condition, image]

        # Set up EDF data file name and local data folder
        #
        # The EDF data filename should not exceed eight alphanumeric characters
        # use ONLY number 0-9, letters, and _ (underscore) in the filename
        edf_fname = 'NFB_TEST'

        # Prompt user to specify an EDF data filename
        # before we open a fullscreen window
        # while True:
        #     # use "raw_input" to get user input if running with Python 2.x
        #     # try:
        #     #     input = input()
        #     # except NameError:
        #     #     print('INPUT ERROR NAME')
        #     #     pass
        #     prompt = '\nSpecify an EDF filename\n' + \
        #              'Filename must not exceed eight alphanumeric characters.\n' + \
        #              'ONLY letters, numbers and underscore are allowed.\n\n--> '
        #     edf_fname = input(prompt)
        #     # strip trailing characters, ignore the '.edf' extension
        #     edf_fname = edf_fname.rstrip().split('.')[0]p
        #
        #     # check if the filename is valid (length <= 8 & no special char)
        #     allowed_char = ascii_letters + digits + '_'
        #     if not all([c in allowed_char for c in edf_fname]):
        #         print('ERROR: Invalid EDF filename')
        #     elif len(edf_fname) > 8:
        #         print('ERROR: EDF filename should not exceed 8 characters')
        #     else:
        #         break

        # Set up a folder to store the EDF data files and the associated resources
        # e.g., files defining the interest areas used in each trial
        results_folder = 'results'
        if not os.path.exists(results_folder): #TODO: make a dialog to get the results path to use here (same as with posner task)
            os.makedirs(results_folder)

        # We download EDF data file from the EyeLink Host PC to the local hard
        # drive at the end of each testing session, here we rename the EDF to
        # include session start date/time
        time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
        session_identifier = edf_fname + time_str

        # create a folder for the current testing session in the "results" folder
        session_folder = os.path.join(results_folder, session_identifier)
        print(f"ATTEMPTING TO CREATE SESSION FOLDER: {session_folder}")
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)
            print(f"CREATED SESSION FOLDER")

        # Step 1: Connect to the EyeLink Host PC
        #
        # The Host IP address, by default, is "100.1.1.1".
        # the "el_tracker" objected created here can be accessed through the Pylink
        # Set the Host PC address to "None" (without quotes) to run the script
        # in "Dummy Mode"
        if dummy_mode:
            el_tracker = pylink.EyeLink(None)
        else:
            try:
                el_tracker = pylink.EyeLink("100.1.1.1")
            except RuntimeError as error:
                print('ERROR CONNECTING:', error)
                pygame.quit()
                # sys.exit()

        # Step 2: Open an EDF data file on the Host PC
        edf_file = edf_fname + ".EDF"
        try:
            el_tracker.openDataFile(edf_file)
        except RuntimeError as err:
            print('ERROR OPENING EDF:', err)
            # close the link if we have one open
            if el_tracker.isConnected():
                el_tracker.close()
            pygame.quit()
            # sys.exit()

        # Add a header text to the EDF file to identify the current experiment name
        # This is OPTIONAL. If your text starts with "RECORDED BY " it will be
        # available in DataViewer's Inspector window by clicking
        # the EDF session node in the top panel and looking for the "Recorded By:"
        # field in the bottom panel of the Inspector.
        preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
        el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

        # Step 3: Configure the tracker
        #
        # Put the tracker in offline mode before we change tracking parameters
        el_tracker.setOfflineMode()

        # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
        # 5-EyeLink 1000 Plus, 6-Portable DUO
        eyelink_ver = 0  # set version to 0, in case running in Dummy mode
        if not dummy_mode:
            vstr = el_tracker.getTrackerVersionString()
            eyelink_ver = int(vstr.split()[-1].split('.')[0])
            # print out some version info in the shell
            print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

        # File and Link data control
        # what eye events to save in the EDF file, include everything by default
        file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        # what eye events to make available over the link, include everything by default
        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
        # what sample data to save in the EDF data file and to make available
        # over the link, include the 'HTARGET' flag to save head target sticker
        # data for supported eye trackers
        if eyelink_ver > 3:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        else:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
        el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
        el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
        el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

        # Optional tracking parameters
        # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
        # if eyelink_ver > 2:
        #     el_tracker.sendCommand("sample_rate 1000")
        # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
        el_tracker.sendCommand("calibration_type = HV5")
        # Set a gamepad button to accept calibration/drift check target
        # You need a supported gamepad/button box that is connected to the Host PC
        el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

        # Step 4: set up a graphics environment for calibration
        #
        # open a Pygame window
        win = None
        if full_screen:
            win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF, display=1)
        else:
            win = pygame.display.set_mode((0, 0), 0)

        scn_width, scn_height = win.get_size()
        pygame.mouse.set_visible(False)  # hide mouse cursor

        # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
        # see the EyeLink Installation Guide, "Customizing Screen Settings"
        el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
        el_tracker.sendCommand(el_coords)

        # Write a DISPLAY_COORDS message to the EDF file
        # Data Viewer needs this piece of info for proper visualization, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
        el_tracker.sendMessage(dv_coords)

        # Configure a graphics environment (genv) for tracker calibration
        self.genv = CalibrationGraphics(el_tracker, win)

        # Set background and foreground colors
        # parameters: foreground_color, background_color
        foreground_color = (0, 0, 0)
        background_color = (128, 128, 128)
        self.genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        #
        # The target could be a "circle" (default) or a "picture",
        # To configure the type of calibration target, set
        # genv.setTargetType to "circle", "picture", e.g.,
        # genv.setTargetType('picture')
        #
        # Use gen.setPictureTarget() to set a "picture" target, e.g.,
        # genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))

        self.genv.setTargetType('circle')

        # Configure the size of the calibration target (in pixels)
        # genv.setTargetSize(24)

        # Beeps to play during calibration, validation and drift correction
        # parameters: target, good, error
        #     target -- sound to play when target moves
        #     good -- sound to play on successful operation
        #     error -- sound to play on failure or interruption
        # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        # e.g., genv.setCalibrationSounds('type.wav', 'qbeep.wav', 'error.wav')
        self.genv.setCalibrationSounds('off', 'off', 'off')


        # Request Pylink to use the Pygame window we opened above for calibration
        print('STARTING CALIBRATION')
        pylink.closeGraphics()
        pylink.openGraphicsEx(self.genv)

        # Step 5: Run the experimental trials
        # define a few helper functions for trial handling
        # Show the task instructions
        task_msg = 'Eyelink Calibration\n'
        if dummy_mode:
            task_msg = task_msg + '\nEyelink Dummy mode ON'
        else:
            task_msg = task_msg + '\nPress ENTER to calibrate tracker'

        # Pygame bug warning
        pygame_warning = '\n\nDue to a bug in Pygame 2, the window may have lost' + \
                         '\nfocus and stopped accepting keyboard inputs.' + \
                         '\nClicking the mouse helps get around this issue.' + \
                         '\nPress Ctrl+C to exit after calibration complete'
        if pygame.__version__.split('.')[0] == '2':
            task_msg = task_msg + pygame_warning

        self.show_message(task_msg, (0, 0, 0), (128, 128, 128))
        self.wait_key([pygame.K_RETURN])
        # skip this step if running the script in Dummy Mode
        if not dummy_mode:
            try:
                el_tracker.doTrackerSetup()
                self.wait_key([pygame.K_ESCAPE])
            except RuntimeError as err:
                print('ERROR DOING SETUP:', err)
                el_tracker.exitCalibration()

    def show_message(self,message, fg_color, bg_color):
        """ show messages on the screen

        message: The message you would like to show on the screen
        fg_color/bg_color: color for the texts and the background screen
        """

        # clear the screen and blit the texts
        win_surf = pygame.display.get_surface()
        win_surf.fill(bg_color)

        scn_w, scn_h = win_surf.get_size()
        message_fnt = pygame.font.SysFont('Arial', 32)
        msgs = message.split('\n')
        for i in range(len(msgs)):
            message_surf = message_fnt.render(msgs[i], True, fg_color)
            w, h = message_surf.get_size()
            msg_y = scn_h / 2 + h / 2 * 2.5 * (i - len(msgs) / 2.0)
            win_surf.blit(message_surf, (int(scn_w / 2 - w / 2), int(msg_y)))

        pygame.display.flip()

    def wait_key(self,key_list, duration=sys.maxsize):
        """ detect and return a keypress, terminate the task if ESCAPE is pressed

        parameters:
        key_list: allowable keys (pygame key constants, e.g., [K_a, K_ESCAPE]
        duration: the maximum time allowed to issue a response (in ms)
                  wait for response 'indefinitely' (with sys.maxsize)
        """

        got_key = False
        # clear all cached events if there are any
        pygame.event.clear()
        t_start = pygame.time.get_ticks()
        resp = [None, t_start, -1]

        while not got_key:
            # check for time out
            if (pygame.time.get_ticks() - t_start) > duration:
                break

            # check key presses
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN:
                    if ev.key in key_list:
                        resp = [pygame.key.name(ev.key),
                                t_start,
                                pygame.time.get_ticks()]
                        got_key = True

                if (ev.type == pygame.KEYDOWN) and (ev.key == pygame.K_c):
                    if ev.mod in [pygame.KMOD_LCTRL, pygame.KMOD_RCTRL, 4160, 4224]:
                        self.terminate_task()
                        pass

        # clear the screen following each keyboard response
        win_surf = pygame.display.get_surface()
        win_surf.fill(self.genv.getBackgroundColor())
        pygame.display.flip()

        return resp

    def terminate_task(self):
        """ Terminate the task gracefully and retrieve the EDF data file

        file_to_retrieve: The EDF on the Host that we would like to download
        win: the current window used by the experimental script
        """

        # disconnect from the tracker if there is an active connection
        el_tracker = pylink.getEYELINK()

        # if el_tracker.isConnected():
        #     # Terminate the current trial first if the task terminated prematurely
        #     error = el_tracker.isRecording()
        #     # if error == pylink.TRIAL_OK:
        #     #     abort_trial()
        #
        #     # Put tracker in Offline mode
        #     el_tracker.setOfflineMode()
        #
        #     # Clear the Host PC screen and wait for 500 ms
        #     el_tracker.sendCommand('clear_screen 0')
        #     pylink.msecDelay(500)
        #
        #     # Close the edf data file on the Host
        #     el_tracker.closeDataFile()
        #
        #     # Show a file transfer message on the screen
        #     msg = 'EDF data is transferring from EyeLink Host PC...'
        #     self.show_message(msg, (0, 0, 0), (128, 128, 128))
        #
        #     # Download the EDF data file from the Host PC to a local data folder
        #     # parameters: source_file_on_the_host, destination_file_on_local_drive
        #     local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        #     try:
        #         el_tracker.receiveDataFile(edf_file, local_edf)
        #     except RuntimeError as error:
        #         print('ERROR:', error)
        #
        #     # Close the link to the tracker.
        #     el_tracker.close()

        # quit pygame and python
        print('QUITTING CALIBRATION')
        pygame.quit()
        # sys.exit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FileSelectorLine()
    window.show()
    sys.exit(app.exec_())
