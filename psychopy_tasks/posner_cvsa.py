from psychopy.gui import DlgFromDict
from psychopy.visual import Window, TextStim, circle
from psychopy.core import Clock, quit
from psychopy.event import Mouse
from psychopy.hardware.keyboard import Keyboard
from psychopy.monitors import Monitor
from psychopy.data import TrialHandler, getDateStr, ExperimentHandler
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import psychopy
import os

TRIAL_REPS = [2,2,2,2]
frameTolerance = 0.001  # how close to onset before 'same' frame
expName = 'posner_task'
### DIALOG BOX ROUTINE ###
exp_info = {'participant': "99", 'session': 'x'}
dlg = DlgFromDict(exp_info)

exp_info['date'] = getDateStr()  # add a simple timestamp
exp_info['expName'] = expName
exp_info['psychopyVersion'] = psychopy.__version__

_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (exp_info['participant'], expName, exp_info['date'])
# An ExperimentHandler isn't essential but helps with data saving
thisExp = ExperimentHandler(name=expName, version='',
    extraInfo=exp_info, runtimeInfo=None,
    originPath='C:\\Users\\2354158T\\Documents\\GitHub\\nfb\\psychopy\\posner_eyelink.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)

# If pressed Cancel, abort!
if not dlg.OK:
    quit()
else:
    # Quit when either the participant nr or age is not filled in
    if not exp_info['participant'] or not exp_info['session']:
        quit()

    else:  # let's star the experiment!
        print(f"Started experiment for participant {exp_info['participant']} "
              f"in session {exp_info['session']}.")

# init the monitor
mon = Monitor('eprime',
                 width=40,
                 distance=60,
                 autoLog=True)
mon.setSizePix((1280, 1024))

# Initialize a fullscreen window with my monitor (HD format) size
# and my monitor specification called "samsung" from the monitor center
win = Window(fullscr=False, monitor=mon)

# Also initialize a mouse, for later
# We'll set it to invisible for now
mouse = Mouse(visible=False)

# Initialize clocks
clock = Clock() #global clock
trial_clock = Clock()

# Initialize Keyboard
kb = Keyboard()

# Init the start screen components
welcome_txt_stim = TextStim(win, text="""Welcome to this experiment!
                                         Press SPACE to start""")
welcome_txt_stim.setAutoDraw(True)

# Init the trial components
fc = circle.Circle(
    win=win,
    name='fc',
    units="deg",
    radius=0.1,
    fillColor='black',
    lineColor='black'
)

left_probe = circle.Circle(
    win=win,
    name='left_probe',
    units="deg",
    radius=3.5,
    fillColor='blue',
    lineColor='white',
    lineWidth=8,
    edges=128,
    pos=[-5, -1]
)
right_probe = circle.Circle(
    win=win,
    name='right_probe',
    units="deg",
    radius=3.5,
    fillColor='blue',
    lineColor='white',
    lineWidth=8,
    edges=256,
    pos=[5, -1]
)

### START BODY OF EXPERIMENT ###
#
# Do the init screen
win.flip()
# Wait until space is pressed before moving on
while True:
    keys = kb.getKeys()
    if "space" in keys:
        break

trials_1 = TrialHandler(nReps=TRIAL_REPS[0], method='sequential',
    extraInfo=exp_info, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials_1)  # add the loop to the experiment
thisTrial = trials_1.trialList[0]  # so we can initialise stimuli with some values

# Do the trials
for trial_index, thisTrial in enumerate(trials_1):
    print(f'STARTING TRIAL: {trials_1.thisN}')
    currentLoop = trials_1
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))

    continueRoutine = True
    cueComponents = [fc, left_probe, right_probe]

    # Reset the trial clock
    trial_clock.reset()

    for thisComponent in cueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    while continueRoutine:
        # get current time
        t = trial_clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trial_clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)

        # Handle both the probes
        if left_probe.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            left_probe.tStart = t  # local t and not account for scr refresh
            left_probe.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_probe, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_probe.started')
            left_probe.setAutoDraw(True)
        if left_probe.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_probe.tStartRefresh + 6.6-frameTolerance:
                left_probe.tStop = t  # not accounting for scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_l_cue.stopped')
                left_probe.setAutoDraw(False)
                left_probe.status = FINISHED

        if right_probe.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            right_probe.tStart = t  # local t and not account for scr refresh
            right_probe.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_probe, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_probe.started')
            right_probe.setAutoDraw(True)
        if right_probe.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right_probe.tStartRefresh + 6.6-frameTolerance:
                right_probe.tStop = t  # not accounting for scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_probe.stopped')
                right_probe.setAutoDraw(False)
                right_probe.status = FINISHED

        # Handle the fixation circle
        if fc.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fc.tStart = t  # local t and not account for scr refresh
            fc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fc.started')
            fc.setAutoDraw(True)
        if fc.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fc.tStartRefresh + 1.0-frameTolerance:
                fc.tStop = t  # not accounting for scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fc.stopped')
                fc.setAutoDraw(False)
                fc.status = FINISHED

        # check for quit (typically the Esc key)
        if kb.getKeys(keyList=["escape"]):
            quit()

        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in cueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "cue" ---
    for thisComponent in cueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.nextEntry()

### END BODY OF EXPERIMENT ###

# Finish experiment by closing window and quitting
win.close()
quit()