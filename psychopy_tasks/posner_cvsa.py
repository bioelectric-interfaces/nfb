from psychopy.gui import DlgFromDict
from psychopy.visual import Window, TextStim, circle
from psychopy.core import Clock, quit
from psychopy.event import Mouse
from psychopy.hardware.keyboard import Keyboard
from psychopy.monitors import Monitor
from psychopy.data import TrialHandler, getDateStr, ExperimentHandler
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
from psychopy.visual.shape import ShapeStim
import psychopy
import os
import typing
import random
import time
from dataclasses import dataclass
import pylink
import sys
import logging

from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


@dataclass
class PosnerComponent:
    component: typing.Any = object()
    start_time: float = 0.0
    duration: float = 1.0
    blocking: bool = False
    name: str = 'component'
    allKeys: list = None
    keyList: list = None
    input_component: bool = False


class PosnerTask:
    def __init__(self):
        self.trial_reps = [1, 1, 1, 1]
        self.frameTolerance = 0.001  # how close to onset before 'same' frame
        self.expName = 'posner_task'
        self.exp_info = {'participant': "99", 'session': 'x'}
        self.thisExp = None

        # init the monitor
        self.mon = Monitor('eprime',
                           width=40,
                           distance=60,
                           autoLog=True)
        self.mon.setSizePix((1280, 1024))
        self.win = Window(fullscr=True, monitor=self.mon, screen=1)
        self.scn_width, self.scn_height = self.win.size

        # init the trial component dicts
        self.start_components = {}
        self.trial_components = {}
        self.continue_components = {}
        self.end_components = {}

        # init the global keyboard
        self.kb = Keyboard()

        # Initialize clocks
        self.global_clock = Clock()
        self.trial_clock = Clock()

        # init component start times
        self.trial_duration = 6.5
        self.fc_duration = 1.0
        self.cue_duration = self.trial_duration - self.fc_duration
        self.stim_duration = 0.1
        self.probe_start_time = random.uniform(self.fc_duration + 3, self.trial_duration - self.stim_duration - 1)

        # init the results paths
        self.session_folder = "results"
        self.edf_file = 'eye_data'

        # timestamp the start of the whole thing
        self.time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
        # timestamp_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')

    def init_eye_link(self):
        dummy_mode = False
        edf_fname = f"{self.exp_info['participant']}_psnr" # TODO: fix the data folders so everything is convinently located
        results_folder = 'eye_track_data'
        eyelink_ip = "100.1.1.1"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        session_identifier = edf_fname + self.time_str
        self.session_folder = os.path.join(results_folder, session_identifier)
        if not os.path.exists(self.session_folder):
            os.makedirs(self.session_folder)

        # Step 1: Connect to the EyeLink Host PC
        if dummy_mode:
            self.el_tracker = pylink.EyeLink(None)
        else:
            try:
                self.el_tracker = pylink.EyeLink(eyelink_ip)
            except RuntimeError as error:
                print('ERROR:', error)
                logging.error(error)
                quit()
                sys.exit()

        # Step 2: Open an EDF data file on the Host PC
        self.edf_file = edf_fname + ".EDF"
        try:
            self.el_tracker.openDataFile(self.edf_file)
        except RuntimeError as err:
            print('ERROR:', err)
            logging.error(err)
            # close the link if we have one open
            if self.el_tracker.isConnected():
                self.el_tracker.close()
            quit()
            sys.exit()
        preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
        self.el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

        # Step 3: Configure the tracker
        self.el_tracker.setOfflineMode()
        eyelink_ver = 0  # set version to 0, in case running in Dummy mode
        if not dummy_mode:
            vstr = self.el_tracker.getTrackerVersionString()
            eyelink_ver = int(vstr.split()[-1].split('.')[0])
            # print out some version info in the shell
            print('Running experiment on %s, version %d' % (vstr, eyelink_ver))
            logging.info('Running experiment on %s, version %d' % (vstr, eyelink_ver))
        else:
            print(f'Running experiment in dummy mode')
            logging.info(f'Running experiment in dummy mode')

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
        self.el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
        self.el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
        self.el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
        self.el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

        # calibration type
        self.el_tracker.sendCommand("calibration_type = HV5")

        # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
        el_coords = "screen_pixel_coords = 0 0 %d %d" % (self.scn_width - 1, self.scn_height - 1)
        self.el_tracker.sendCommand(el_coords)

        # Write a DISPLAY_COORDS message to the EDF file
        # Data Viewer needs this piece of info for proper visualization, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (self.scn_width - 1, self.scn_height - 1)
        self.el_tracker.sendMessage(dv_coords)

        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.el_tracker, self.win)
        print(genv)  # print out the version number of the CoreGraphics library
        logging.info(genv)

        # Set background and foreground colors for the calibration target
        foreground_color = (-1, -1, -1)
        background_color = self.win.color
        genv.setCalibrationColors(foreground_color, background_color)

        # Set up the calibration target
        genv.setTargetType('circle')
        genv.setCalibrationSounds('off', 'off', 'off')

        # Request Pylink to use the PsychoPy window we opened above for calibration
        print(f"STARTING CALIBRARION")
        logging.info(f"STARTING CALIBRARION")
        pylink.openGraphicsEx(genv)

        # Step 5: Set up the camera and calibrate the tracker
        ## Show the task instructions
        task_msg = 'Eyelink Calibration\n'
        if dummy_mode:
            task_msg = task_msg + '\nEyelink Dummy mode ON'
        else:
            task_msg = task_msg + '\nPress ENTER to calibrate tracker'
        self.show_msg(self.win, task_msg, genv)

    def calibrate_eyelink(self, dummy_mode=False):
        if not dummy_mode:
            try:
                self.el_tracker.doTrackerSetup()
            except RuntimeError as err:
                print('ERROR:', err)
                logging.error('ERROR:', err)
                self.el_tracker.exitCalibration()
        print(f"CALIBRARION DONE")
        logging.info(f"CALIBRARION DONE")

    def show_msg(self, win, text, genv, wait_for_keypress=True):
        """ Show task instructions on screen"""

        msg = TextStim(win, text,
                       color=genv.getForegroundColor(),
                       wrapWidth=self.scn_width / 2)
        self.clear_screen(win, genv)
        msg.draw()
        win.flip()

        # wait indefinitely, terminates upon any key press
        if wait_for_keypress:
            psychopy.event.waitKeys()
            self.clear_screen(win, genv)

    def clear_screen(self, win, genv):
        """ clear up the PsychoPy window"""
        win.fillColor = genv.getBackgroundColor()
        win.flip()

    def update_exp_info(self):
        self.exp_info['date'] = getDateStr()  # add a simple timestamp
        self.exp_info['expName'] = self.expName
        self.exp_info['psychopyVersion'] = psychopy.__version__

    def set_experiment(self):
        _thisDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(_thisDir)
        # setup logging
        output_dir = 'data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(_thisDir, output_dir, f"{self.exp_info['participant']}_{self.expName}_{self.exp_info['date']}")
        # filename = _thisDir + os.sep + u'data/%s_%s_%s' % (
        #     self.exp_info['participant'], self.expName, self.exp_info['date'])
        # An ExperimentHandler isn't essential but helps with data saving
        self.thisExp = ExperimentHandler(name=self.expName, version='',
                                         extraInfo=self.exp_info, runtimeInfo=None,
                                         originPath='C:\\Users\\2354158T\\Documents\\GitHub\\nfb\\psychopy\\posner_eyelink.py',
                                         savePickle=True, saveWideText=True,
                                         dataFileName=filename)
        logging.basicConfig(filename=os.path.join(output_dir, f"{self.exp_info['participant']}_{self.expName}_{self.exp_info['date']}.log"), level=logging.INFO,
                            filemode='w') # TODO: fix log file not saving
        logging.info(f"log created at: {self.time_str}")
        log_file = os.path.join(output_dir, f"{self.exp_info['participant']}_{self.expName}_{self.exp_info['date']}.log")
        print(f"LOG CREATED: {log_file}")

    def calculate_cue_side(self):
        """
        Calculate the direction the cue will be
        side: direction of the cue (1 = left, 2 = right, 3 = centre
        cue probability is equal for left, right, and centre
        """
        self.probe_start_time = random.uniform(self.fc_duration + 3, self.trial_duration - self.stim_duration - 1)
        cue_dir = random.choice([1, 2, 3])  # 1=l, 2=r, 3=n

        self.left_cue.component.opacity = 0.0
        self.right_cue.component.opacity = 0.0
        self.centre_cue1.component.opacity = 0.0
        self.centre_cue2.component.opacity = 0.0
        if cue_dir == 1:
            self.left_cue.component.opacity = 1.0
        elif cue_dir == 2:
            self.right_cue.component.opacity = 1.0
        elif cue_dir == 3:
            self.centre_cue1.component.opacity = 1.0
            self.centre_cue2.component.opacity = 1.0
        return cue_dir

    def calculate_stim_validity(self, cue_dir, valid_cue_weight=70):
        """
        Calculate if a stimulation is valid or not
        cue_dir: direction of cue
        valid_cue_weight: chance the stim is valid
        """
        neutral_side = random.choice([1, 2])  # side to display stim in case of neutral/centre cue
        valid_cue = random.choices([True, False], weights=(valid_cue_weight, 100 - valid_cue_weight))[0]
        if valid_cue:
            # valid cue
            if cue_dir == 1:
                stim_pos = (-5, -1)
            elif cue_dir == 2:
                stim_pos = (5, -1)
            elif cue_dir == 3:
                # 50% chance left or right
                if neutral_side == 1:
                    stim_pos = (-5, -1)
                elif neutral_side == 2:
                    stim_pos = (5, -1)
        else:
            # invalid cue
            if cue_dir == 1:
                stim_pos = (5, -1)
            elif cue_dir == 2:
                stim_pos = (-5, -1)
            elif cue_dir == 3:
                # 50% chance left or right
                if neutral_side == 1:
                    stim_pos = (-5, -1)
                elif neutral_side == 2:
                    stim_pos = (5, -1)
        self.stim.component.setPos(stim_pos)
        return valid_cue

    def init_start_components(self):
        self.start_text = PosnerComponent(
            TextStim(self.win, text="""Welcome to this experiment!
                                                 Press DOWN arrow to start"""),
            duration=0.0,
            blocking=True,
            input_component=True)
        self.start_components = {'start_text': self.start_text}

    def init_continue_components(self):
        self.continue_text = PosnerComponent(
            TextStim(self.win, text="""you've finished X blocks
                                                 Press DOWN arrow to continue"""),
            duration=0.0,
            blocking=True,
            input_component=True)
        self.continue_components = {'continue_text': self.continue_text}

    def init_end_components(self):
        self.end_text = PosnerComponent(
            TextStim(self.win, text="""you've finished!"""),
            duration=0.0,
            blocking=True,
            input_component=True)
        self.end_components = {'end_text': self.end_text}

    def init_trial_components(self):
        self.fc = PosnerComponent(
            circle.Circle(
                win=self.win,
                units="deg",
                radius=0.1,
                fillColor='black',
                lineColor='black'
            ),
            duration=self.fc_duration,
            start_time=0.0)

        self.left_probe = PosnerComponent(
            circle.Circle(
                win=self.win,
                units="deg",
                radius=3.5 / 2,
                fillColor='blue',
                lineColor='white',
                lineWidth=8,
                edges=128,
                pos=[-5, -1],
            ),
            duration=self.trial_duration,
            start_time=0.0)

        self.right_probe = PosnerComponent(
            circle.Circle(
                win=self.win,
                units="deg",
                radius=3.5 / 2,
                fillColor='blue',
                lineColor='white',
                lineWidth=8,
                edges=256,
                pos=[5, -1],
            ),
            duration=self.trial_duration,
            start_time=0.0)

        self.stim = PosnerComponent(
            circle.Circle(
                win=self.win,
                units="deg",
                radius=0.5,
                fillColor='white',
                lineColor='white',
                edges=256,
                pos=[-5, -1],
            ),
            duration=self.stim_duration,
            start_time=0)

        self.left_cue = PosnerComponent(
            ShapeStim(
                win=self.win, units='deg',
                size=(0.75, 0.75), vertices='triangle',
                ori=-90.0, pos=(0, 0), anchor='center',
                lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
                opacity=1.0, interpolate=True),
            duration=self.cue_duration,
            start_time=self.fc_duration)

        self.right_cue = PosnerComponent(
            ShapeStim(
                win=self.win, units='deg',
                size=(0.75, 0.75), vertices='triangle',
                ori=90.0, pos=(0, 0), anchor='center',
                lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
                opacity=0.0, interpolate=True),
            duration=self.cue_duration,
            start_time=self.fc_duration)

        self.centre_cue1 = PosnerComponent(
            ShapeStim(
                win=self.win, units='deg',
                size=(0.75, 0.75), vertices='triangle',
                ori=90.0, pos=(0.375, 0), anchor='center',
                lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
                opacity=0.0, interpolate=True),
            duration=self.cue_duration,
            start_time=self.fc_duration)

        self.centre_cue2 = PosnerComponent(
            ShapeStim(
                win=self.win, units='deg',
                size=(0.75, 0.75), vertices='triangle',
                ori=-90.0, pos=(-0.375, 0), anchor='center',
                lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
                opacity=0.0, interpolate=True),
            duration=self.cue_duration,
            start_time=self.fc_duration)

        self.key_resp = PosnerComponent(
            Keyboard(),
            duration=1.0,
            start_time=0.0,
            allKeys=[],
            keyList=['left', 'right'])

        self.key_log = PosnerComponent(
            Keyboard(),
            duration=self.trial_duration,
            start_time=0.0,
            allKeys=[],
            keyList=['left', 'right', 'up', 'down'])

        self.trial_components = {'fc': self.fc,
                                 'left_probe': self.left_probe,
                                 'right_probe': self.right_probe,
                                 'left_cue': self.left_cue,
                                 'right_cue': self.right_cue,
                                 'centre_cue1': self.centre_cue1,
                                 'centre_cue2': self.centre_cue2,
                                 'stim': self.stim,
                                 'key_resp': self.key_resp,
                                 'key_log': self.key_log}

    def handle_component(self, pcomp, pcomp_name, tThisFlip, tThisFlipGlobal, t, trial_id, duration=1):
        # Handle both the probes
        # el_tracker.sendMessage( f'TTHISFLIP {tThisFlip}, TTHISFLIPGLOBAL {tThisFlipGlobal} {pcomp.name}_START_TIME {pcomp.start_time}')
        waitOnFlip = False
        if pcomp.component.status == NOT_STARTED and tThisFlip >= pcomp.start_time - self.frameTolerance:  # TODO: it's possible that the start_time doesn't take into account the flip which is where the descrepency comes from
            # keep track of start time/frame for later
            pcomp.component.tStart = t  # local t and not account for scr refresh
            pcomp.component.tStartRefresh = tThisFlipGlobal  # on global time
            self.win.timeOnFlip(pcomp.component, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            self.thisExp.timestampOnFlip(self.win, f'{pcomp_name}.started')
            if hasattr(pcomp.component, "setAutoDraw"):
                pcomp.component.setAutoDraw(True)
            if isinstance(pcomp.component, Keyboard):
                pcomp.component.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                self.win.callOnFlip(pcomp.component.clock.reset)  # t=0 on next screen flip
                self.win.callOnFlip(pcomp.component.clearEvents,
                                    eventType='keyboard')  # clear events on next screen flip
            self.win.callOnFlip(self.el_tracker.sendMessage, f'TRIAL_{trial_id}_{pcomp_name}_START')
        if pcomp.component.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pcomp.component.tStartRefresh + duration - self.frameTolerance:
                if not pcomp.blocking:
                    pcomp.component.tStop = t  # not accounting for scr refresh
                    # add timestamp to datafile
                    self.thisExp.timestampOnFlip(self.win, f'{pcomp_name}.stopped')
                    if hasattr(pcomp.component, "setAutoDraw"):
                        pcomp.component.setAutoDraw(False)
                    pcomp.component.status = FINISHED
                    self.win.callOnFlip(self.el_tracker.sendMessage, f'TRIAL_{trial_id}_{pcomp_name}_END')
        if isinstance(pcomp.component, Keyboard):
            if pcomp.component.status == STARTED and not waitOnFlip:
                theseKeys = pcomp.component.getKeys(keyList=pcomp.keyList, waitRelease=False)
                # self._key_resp_allKeys.extend(theseKeys)
                pcomp.allKeys.extend(theseKeys)
                if len(pcomp.allKeys):
                    pcomp.component.keys = [key.name for key in pcomp.allKeys]  # storing all keys
                    pcomp.component.rt = [key.rt for key in pcomp.allKeys]
                    # print(pcomp.component.keys)
                    # print(pcomp.component.rt)

    def eye_tracker_trial_setup(self):
        # get a reference to the currently active EyeLink connection
        el_tracker = pylink.getEYELINK()

        # put the tracker in the offline mode first
        el_tracker.setOfflineMode()

        # clear the host screen before we draw the backdrop
        el_tracker.sendCommand('clear_screen 0')

        # OPTIONAL: draw landmarks and texts on the Host screen
        left = int(self.scn_width / 2.0) - 60  # TODO: make this appropriate to the posner task
        top = int(self.scn_height / 2.0) - 60
        right = int(self.scn_width / 2.0) + 60
        bottom = int(self.scn_height / 2.0) + 60
        draw_cmd = 'draw_filled_box %d %d %d %d 1' % (left, top, right, bottom)
        el_tracker.sendCommand(draw_cmd)
        return el_tracker

    def start_eye_tracker_recording(self, el_tracker):
        # put tracker in idle/offline mode before recording
        el_tracker.setOfflineMode()

        # Start recording
        # arguments: sample_to_file, events_to_file, sample_over_link,
        # event_over_link (1-yes, 0-no)
        try:
            el_tracker.startRecording(1, 1, 1, 1)
        except RuntimeError as error:
            print("ERROR:", error)
            logging.error(error)
            el_tracker.sendMessage(f'ERROR {error}')
            self.abort_trial(el_tracker)
            return pylink.TRIAL_ERROR
        # Allocate some time for the tracker to cache some samples
        pylink.pumpDelay(100)
        return el_tracker

    def eye_tracker_drift_correction(self, el_tracker, dummy_mode=False):
        # drift check
        # we recommend drift-check at the beginning of each trial
        # the doDriftCorrect() function requires target position in integers
        # the last two arguments:
        # draw_target (1-default, 0-draw the target then call doDriftCorrect)
        # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
        #
        # Skip drift-check if running the script in Dummy Mode
        print('STARTING DRIFT CORRECTION')
        logging.info('STARTING DRIFT CORRECTION')
        if el_tracker.isConnected():
            # Terminate the current trial first if the task terminated prematurely
            error = el_tracker.isRecording()
            if error == pylink.TRIAL_OK:
                _, el_tracker = self.abort_trial(el_tracker)

            # Put tracker in Offline mode
            el_tracker.setOfflineMode()

            # Clear the Host PC screen and wait for 500 ms
            el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)
        while not dummy_mode:
            # terminate the task if no longer connected to the tracker or
            # user pressed Ctrl-C to terminate the task
            if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                # self.terminate_task()
                return pylink.ABORT_EXPT

            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                print('DRIFT STARTED')
                logging.info('DRIFT STARTED')
                error = el_tracker.doDriftCorrect(int(self.scn_width / 2.0),
                                                  int(self.scn_height / 2.0), 1, 0)
                # break following a success drift-check
                print('DRIFT FINISHED')
                logging.info('DRIFT FINISHED')
                if error is not pylink.ESC_KEY:
                    print('DRIFT BREAK')
                    logging.info('DRIFT BREAK')
                    break
            except:
                pass
        return el_tracker

    def abort_trial(self, el_tracker):
        """Ends eyetracker recording """
        el_tracker = pylink.getEYELINK()

        # Stop recording
        if el_tracker.isRecording():
            # add 100 ms to catch final trial events
            pylink.pumpDelay(100)
            el_tracker.stopRecording()

        # Send a message to clear the Data Viewer screen
        bgcolor_RGB = (116, 116, 116)
        el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

        # send a message to mark trial end
        el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

        return pylink.TRIAL_ERROR, el_tracker

    def run_block(self, component_dict, trial_reps, block_name='block', block_id=0):
        trials = TrialHandler(nReps=trial_reps, method='sequential',
                              extraInfo=self.exp_info, originPath=-1,
                              trialList=[None],
                              seed=None, name='trials')
        self.thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values

        # update continue text
        if 'continue' in block_name:
            component_dict['continue_text'].component.setText(f"""you've finished {block_id+1}/4 blocks
                                                 Press DOWN arrow to continue""")

        # Do the trials
        for trial_index, thisTrial in enumerate(trials):
            trial_id = f'{block_id}-{trial_index}'
            print(f'STARTING TRIAL: {trial_id}, thisN: {trials.thisN} OF BLOCK: {block_name}')
            logging.info(f'STARTING TRIAL: {trial_id}, thisN: {trials.thisN} OF BLOCK: {block_name}')

            # Calculate the side of the cue and stim validity
            cue_dir = self.calculate_cue_side()
            valid_cue = self.calculate_stim_validity(cue_dir=cue_dir)

            # Set the stim start time
            if 'stim' in component_dict:
                component_dict['stim'].start_time = self.probe_start_time

            # Set the keyboard response duration
            if 'key_resp' in component_dict:
                component_dict['key_resp'].start_time = self.probe_start_time
                component_dict[
                    'key_resp'].duration = self.trial_duration - self.probe_start_time

            # record_status_message : show some info on the Host PC
            # here we show how many trial has been tested
            cue_dict = {1: 'LEFT', 2: 'RIGHT', 3: 'CENTRE'}
            status_msg = f'TRIAL {block_name}_{trial_id}: {cue_dict[cue_dir]}'
            self.el_tracker.sendCommand("record_status_message '%s'" % status_msg)

            currentLoop = trials
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    exec('{} = thisTrial[paramName]'.format(paramName))

            continueRoutine = True

            for component_name, component in component_dict.items():
                component.component.tStart = None
                component.component.tStop = None
                component.component.tStartRefresh = None
                component.component.tStopRefresh = None
                if hasattr(component.component, 'status'):
                    component.component.status = NOT_STARTED
                if component.input_component:
                    component.blocking = True

                if isinstance(component.component, Keyboard):
                    # Setup key response
                    component.component.keys = []
                    component.component.rt = []
                    component.allKeys = []
                    self._key_resp_allKeys = []
                    self._key_log_allKeys = []

            # send a "TRIALID" message to mark the start of a trial, see Data
            self.win.callOnFlip(self.el_tracker.sendMessage, f'TRIAL_{trial_id}_BLOCK:{block_name}_START')

            # Reset the trial clock
            self.trial_clock.reset()
            tThisFlipStart = self.win.getFutureFlipTime(clock=self.trial_clock)
            while continueRoutine:
                # get current time
                t = self.trial_clock.getTime()
                tThisFlip = self.win.getFutureFlipTime(clock=self.trial_clock)
                tThisFlipGlobal = self.win.getFutureFlipTime(clock=None)

                # Handle the components
                for component_name, component in component_dict.items():
                    self.handle_component(pcomp=component, pcomp_name=component_name, tThisFlip=tThisFlip,
                                          tThisFlipGlobal=tThisFlipGlobal, t=t,
                                          trial_id=trial_id,
                                          duration=component.duration)
                    # check for blocking end (typically the Space key)
                    if self.kb.getKeys(keyList=["space", 'down']):
                        component.blocking = False

                # check for quit (typically the Esc key)
                if self.kb.getKeys(keyList=["escape"]):
                    self.end_experiment()

                continueRoutine = False  # will revert to True if at least one component still running
                for component_name, component in component_dict.items():
                    if hasattr(component.component, "status") and component.component.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    self.win.flip()

            # --- Ending Routine "cue" ---
            for component_name, component in component_dict.items():
                if hasattr(component.component, "setAutoDraw"):
                    component.component.setAutoDraw(False)
                if isinstance(component.component, Keyboard):
                    # check responses
                    print(f'{component_name}: {component.component.keys}')
                    print(f'{component_name}: {component.component.rt}')
                    if component.component.keys in ['', [], None]:  # No response was made
                        component.component.keys = None
                    trials.addData(f'{component_name}.keys', component.component.keys)
                    if component.component.keys != None:  # we had a response
                        trials.addData(f'{component_name}.rt', component.component.rt)

            # Save extra data
            self.thisExp.addData('block_name', block_name)
            self.thisExp.addData('bloack_id', block_id)
            self.thisExp.addData('cue_dir', cue_dir)
            self.thisExp.addData('valid_cue', valid_cue)

            self.thisExp.nextEntry()

            self.win.callOnFlip(self.el_tracker.sendMessage, f'TRIAL_{trial_id}_BLOCK:{block_name}_END')

    def show_start_dialog(self):
        dlg = DlgFromDict(self.exp_info)
        # If pressed Cancel, abort!
        if not dlg.OK:
            quit()
        else:
            # Quit when either the participant nr or age is not filled in
            if not self.exp_info['participant'] or not self.exp_info['session']:
                quit()

            else:  # let's star the experiment!
                print(f"Started experiment for participant {self.exp_info['participant']} "
                      f"in session {self.exp_info['session']}.")
                logging.info(f"Started experiment for participant {self.exp_info['participant']} "
                      f"in session {self.exp_info['session']}.")

    def end_experiment(self):
        """ Terminate the task gracefully and retrieve the EDF data file

        file_to_retrieve: The EDF on the Host that we would like to download
        win: the current window used by the experimental script
        """
        el_tracker = pylink.getEYELINK()
        if el_tracker.isConnected():
            # Terminate the current trial first if the task terminated prematurely
            error = el_tracker.isRecording()
            if error == pylink.TRIAL_OK:
                self.abort_trial(el_tracker)
            # Put tracker in Offline mode
            el_tracker.setOfflineMode()
            # Clear the Host PC screen and wait for 500 ms
            el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)
            # Close the edf data file on the Host
            el_tracker.closeDataFile()

            # Show a file transfer message on the screen
            msg = 'EDF data is transferring from EyeLink Host PC...'
            print(msg)
            logging.info(msg)

            # Download the EDF data file from the Host PC to a local data folder
            # parameters: source_file_on_the_host, destination_file_on_local_drive
            local_edf = os.path.join(
                self.session_folder + '.EDF')  # TODO: make sure this filename is ok (without the session_identifier)
            print(f'EDF FILE PATH: {local_edf}')
            logging.info(f'EDF FILE PATH: {local_edf}')
            try:
                el_tracker.receiveDataFile(self.edf_file, local_edf)
            except RuntimeError as error:
                print('ERROR:', error)
                logging.error('ERROR:', error)

            # Close the link to the tracker.
            el_tracker.close()

        # close the PsychoPy window
        self.win.close()

        logging.shutdown()

        # Finish experiment by closing window and quitting
        self.win.close()
        quit()

    def run_experiment(self):
        dummy_mode = True
        self.show_start_dialog()
        self.update_exp_info()
        self.set_experiment()
        self.init_start_components()
        self.init_continue_components()
        self.init_end_components()
        self.init_trial_components()
        self.init_eye_link()
        self.calibrate_eyelink()
        el_tracker = self.eye_tracker_trial_setup()
        self.el_tracker = self.start_eye_tracker_recording(el_tracker)
        self.run_block(self.start_components, 1, block_name='start')
        for blockN in range(4):
            self.run_block(self.trial_components, self.trial_reps[0], block_name='trials', block_id=blockN)
            self.run_block(self.continue_components, 1, block_name='continue', block_id=blockN)
            self.el_tracker = self.eye_tracker_drift_correction(self.el_tracker, dummy_mode=dummy_mode)
            self.el_tracker = self.start_eye_tracker_recording(el_tracker)
        self.end_experiment()


if __name__ == '__main__':
    pt = PosnerTask()
    pt.run_experiment()
