#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.1.2),
    on May 16, 2022, at 17:53
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from pylsl import StreamInlet, resolve_stream
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

#streams = resolve_stream('name', 'NFBLab_data1')
#
## create a new inlet to read from the stream
#inlet = StreamInlet(streams[0])
#stream_info_xml = inlet.info().as_xml()
#rt = ET.fromstring(stream_info_xml)
#channels_tree = rt.find('desc').findall("channel") or rt.find('desc').find("channels").findall(
#    "channel")
#labels = [(ch.find('label') if ch.find('label') is not None else ch.find('name')).text
#          for ch in channels_tree]
#          
#aai_idx = labels.index('AAI')
#print(labels)
#print(aai_idx)

#chunk, timestamps = inlet.pull_chunk()
#logging.log(level=logging.INFO, msg='CHUNK1')
#print("CHUNK1:")
#print(chunk)

def rescale(sample, un_scaled_min=0, un_scaled_max=1, scaled_min=255, scaled_max=0):
    un_scaled_range = (un_scaled_max - un_scaled_min)
    scaled_range = (scaled_max - scaled_min)
    scaled_sample =(((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
    scaled_sample = np.clip(scaled_sample, 0, 255)
    return scaled_sample



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.1.2'
expName = 'posner_test'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\2354158T\\Documents\\psychopy_experiments\\posner_test.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=(1024, 768), fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Setup ioHub
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# Initialize components for Routine "start"
startClock = core.Clock()
start_instructions = visual.TextStim(win=win, name='start_instructions',
    text='This is the start of the experiment.\npress something',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
start_resp = keyboard.Keyboard()

# Initialize components for Routine "cue"
cueClock = core.Clock()
import random, copy
from random import randint
from decimal import *

# setup an array to store 6 locations to randomly allocate to our shapes 
# (or select random locations between (-1,-1) and (1,1) if you want them completely 
# random though you then need to factor in overlaying shapes)
master_positions=[[-5,0], [5,0]]
positions = copy.deepcopy(master_positions)
left_cue = visual.ShapeStim(
    win=win, name='left_cue',units='cm', 
    size=(0.5, 0.5), vertices='triangle',
    ori=-90.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
right_cue = visual.ShapeStim(
    win=win, name='right_cue',units='cm', 
    size=(0.5, 0.5), vertices='triangle',
    ori=90.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
fc = visual.ShapeStim(
    win=win, name='fc', vertices='cross',units='cm', 
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)

# Initialize components for Routine "trial"
trialClock = core.Clock()
fixation_cross = visual.ShapeStim(
    win=win, name='fixation_cross', vertices='cross',units='cm', 
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
probe_l = visual.ShapeStim(
    win=win, name='probe_l',units='cm', 
    size=(2, 2), vertices='circle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=5.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-2.0, interpolate=True)
probe_r = visual.ShapeStim(
    win=win, name='probe_r',units='cm', 
    size=(2, 2), vertices='circle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=5.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, 1.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-3.0, interpolate=True)
colour_stim = visual.ShapeStim(
    win=win, name='colour_stim',units='cm', 
    size=(2, 2), vertices='circle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-4.0, interpolate=True)
probe_r_fill = visual.ShapeStim(
    win=win, name='probe_r_fill',units='cm', 
    size=(1.5, 1.5), vertices='circle',
    ori=0.0, pos=(10,-2.5), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
probe_l_fill = visual.ShapeStim(
    win=win, name='probe_l_fill',units='cm', 
    size=(1.5, 1.5), vertices='circle',
    ori=0.0, pos=(-10,-2.5), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-6.0, interpolate=True)
text = visual.TextStim(win=win, name='text',
    text='',
    font='Open Sans',
    units='cm', pos=(0, 4), height=1.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-7.0);
key_resp = keyboard.Keyboard()
text2 = visual.TextStim(win=win, name='text2',
    text='',
    font='Open Sans',
    units='cm', pos=(0, 3), height=1.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-9.0);
key_log = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "start"-------
continueRoutine = True
# update component parameters for each repeat
start_resp.keys = []
start_resp.rt = []
_start_resp_allKeys = []
# keep track of which components have finished
startComponents = [start_instructions, start_resp]
for thisComponent in startComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
startClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "start"-------
while continueRoutine:
    # get current time
    t = startClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=startClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *start_instructions* updates
    if start_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        start_instructions.frameNStart = frameN  # exact frame index
        start_instructions.tStart = t  # local t and not account for scr refresh
        start_instructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(start_instructions, 'tStartRefresh')  # time at next scr refresh
        start_instructions.setAutoDraw(True)
    
    # *start_resp* updates
    waitOnFlip = False
    if start_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        start_resp.frameNStart = frameN  # exact frame index
        start_resp.tStart = t  # local t and not account for scr refresh
        start_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(start_resp, 'tStartRefresh')  # time at next scr refresh
        start_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(start_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(start_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if start_resp.status == STARTED and not waitOnFlip:
        theseKeys = start_resp.getKeys(keyList=['space'], waitRelease=False)
        _start_resp_allKeys.extend(theseKeys)
        if len(_start_resp_allKeys):
            start_resp.keys = _start_resp_allKeys[-1].name  # just the last key pressed
            start_resp.rt = _start_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in startComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "start"-------
for thisComponent in startComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('start_instructions.started', start_instructions.tStartRefresh)
thisExp.addData('start_instructions.stopped', start_instructions.tStopRefresh)
# check responses
if start_resp.keys in ['', [], None]:  # No response was made
    start_resp.keys = None
thisExp.addData('start_resp.keys',start_resp.keys)
if start_resp.keys != None:  # we had a response
    thisExp.addData('start_resp.rt', start_resp.rt)
thisExp.addData('start_resp.started', start_resp.tStartRefresh)
thisExp.addData('start_resp.stopped', start_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "start" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=10.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "cue"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    import random 
    probe_start = random.uniform(2.5, 3)
    probe_location = (random.choice([-5, 5]), 0)
    # reset 'positions' 
    positions = copy.deepcopy(master_positions) 
    
    #randomise this for each trial
    random.shuffle(positions)
    print(probe_start)
    
    side = random.choice([0,1])
    stim_side = random.choice([0,1])
    if side == 0:
        r_fill_color = (0,0,0,0)
        l_fill_color = (255,255,255)
        l_cue_color = (255,255,255)
        r_cue_color = (0,0,0,0)
    elif side == 1:
        l_fill_color = (0,0,0,0)
        r_fill_color = (255,255,255)
        r_cue_color = (255,255,255)
        l_cue_color = (0,0,0,0)
        
    
    if stim_side == 0:
        r_fill_color = (0,0,0,0)
        l_fill_color = (255,255,255)
    elif stim_side == 1:
        l_fill_color = (0,0,0,0)
        r_fill_color = (255,255,255)
    # keep track of which components have finished
    cueComponents = [left_cue, right_cue, fc]
    for thisComponent in cueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    cueClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "cue"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = cueClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=cueClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        import random 
        import numpy as np
        
        
        r = random.randrange(0, 255)
        #g = random.randrange(0, 255)
        #b = random.randrange(0, 255)
        #print(f"colour = {g}")
        
        r = 127* np.sin(t*2*pi)+127
        cir_pos = ( sin(t*2*pi), cos(t*2*pi) )
        y_pos = 5* np.sin(t)
        cir_color = (r,0,0)
        
        
        
        # *left_cue* updates
        if left_cue.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            left_cue.frameNStart = frameN  # exact frame index
            left_cue.tStart = t  # local t and not account for scr refresh
            left_cue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_cue, 'tStartRefresh')  # time at next scr refresh
            left_cue.setAutoDraw(True)
        if left_cue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_cue.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                left_cue.tStop = t  # not accounting for scr refresh
                left_cue.frameNStop = frameN  # exact frame index
                win.timeOnFlip(left_cue, 'tStopRefresh')  # time at next scr refresh
                left_cue.setAutoDraw(False)
        if left_cue.status == STARTED:  # only update if drawing
            left_cue.setFillColor(l_cue_color, log=False)
            left_cue.setLineColor(l_cue_color, log=False)
        
        # *right_cue* updates
        if right_cue.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            right_cue.frameNStart = frameN  # exact frame index
            right_cue.tStart = t  # local t and not account for scr refresh
            right_cue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_cue, 'tStartRefresh')  # time at next scr refresh
            right_cue.setAutoDraw(True)
        if right_cue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right_cue.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                right_cue.tStop = t  # not accounting for scr refresh
                right_cue.frameNStop = frameN  # exact frame index
                win.timeOnFlip(right_cue, 'tStopRefresh')  # time at next scr refresh
                right_cue.setAutoDraw(False)
        if right_cue.status == STARTED:  # only update if drawing
            right_cue.setFillColor(r_cue_color, log=False)
            right_cue.setLineColor(r_cue_color, log=False)
        
        # *fc* updates
        if fc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fc.frameNStart = frameN  # exact frame index
            fc.tStart = t  # local t and not account for scr refresh
            fc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fc, 'tStartRefresh')  # time at next scr refresh
            fc.setAutoDraw(True)
        if fc.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fc.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                fc.tStop = t  # not accounting for scr refresh
                fc.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fc, 'tStopRefresh')  # time at next scr refresh
                fc.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in cueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "cue"-------
    for thisComponent in cueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('cue', side)
    thisExp.addData('stim_side', stim_side)
    thisExp.addData('probe_start', probe_start)
    trials.addData('left_cue.started', left_cue.tStartRefresh)
    trials.addData('left_cue.stopped', left_cue.tStopRefresh)
    trials.addData('right_cue.started', right_cue.tStartRefresh)
    trials.addData('right_cue.stopped', right_cue.tStopRefresh)
    trials.addData('fc.started', fc.tStartRefresh)
    trials.addData('fc.stopped', fc.tStopRefresh)
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    # update component parameters for each repeat
    probe_l.setPos((-10, -2.5))
    probe_r.setPos((10, -2.5))
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    key_log.keys = []
    key_log.rt = []
    _key_log_allKeys = []
    # keep track of which components have finished
    trialComponents = [fixation_cross, probe_l, probe_r, colour_stim, probe_r_fill, probe_l_fill, text, key_resp, text2, key_log]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        rnd = random.randrange(0, 255)
        l_color = (0,0,0)
        aai_chunk = 0
        aai_chunk_scaled = 0
        aai_mean = 0
        aai_chunk_scaled = 0
        #try:
        #    chunk, timestamps = inlet.pull_chunk()
        #    logging.log(level=logging.INFO, msg='CHUNK')
        #    print(f"len: {len(chunk)}")
        #    print(f"CHUNK:")
        #    print(chunk)
        #    print("AAI_CHUNK")
        #    aai_chunk = pd.DataFrame(chunk)[4]
        #    print(aai_chunk)
        #    aai_mean = aai_chunk.mean()
        #    print(aai_mean)
        #    aai_chunk_scaled = rescale(aai_mean, un_scaled_min=0, un_scaled_max=0.25)
        #
        #    l_color = (aai_chunk_scaled, aai_chunk_scaled, 255)
        #except:
        #    print("ExcepT")
        #    
        #    l_color = (255, 255, 255)
        #    pass
        
        # *fixation_cross* updates
        if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_cross.frameNStart = frameN  # exact frame index
            fixation_cross.tStart = t  # local t and not account for scr refresh
            fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
            fixation_cross.setAutoDraw(True)
        if fixation_cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_cross.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                fixation_cross.tStop = t  # not accounting for scr refresh
                fixation_cross.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fixation_cross, 'tStopRefresh')  # time at next scr refresh
                fixation_cross.setAutoDraw(False)
        
        # *probe_l* updates
        if probe_l.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            probe_l.frameNStart = frameN  # exact frame index
            probe_l.tStart = t  # local t and not account for scr refresh
            probe_l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(probe_l, 'tStartRefresh')  # time at next scr refresh
            probe_l.setAutoDraw(True)
        if probe_l.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > probe_l.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                probe_l.tStop = t  # not accounting for scr refresh
                probe_l.frameNStop = frameN  # exact frame index
                win.timeOnFlip(probe_l, 'tStopRefresh')  # time at next scr refresh
                probe_l.setAutoDraw(False)
        if probe_l.status == STARTED:  # only update if drawing
            probe_l.setLineColor('red', log=False)
        
        # *probe_r* updates
        if probe_r.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            probe_r.frameNStart = frameN  # exact frame index
            probe_r.tStart = t  # local t and not account for scr refresh
            probe_r.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(probe_r, 'tStartRefresh')  # time at next scr refresh
            probe_r.setAutoDraw(True)
        if probe_r.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > probe_r.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                probe_r.tStop = t  # not accounting for scr refresh
                probe_r.frameNStop = frameN  # exact frame index
                win.timeOnFlip(probe_r, 'tStopRefresh')  # time at next scr refresh
                probe_r.setAutoDraw(False)
        
        # *colour_stim* updates
        if colour_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            colour_stim.frameNStart = frameN  # exact frame index
            colour_stim.tStart = t  # local t and not account for scr refresh
            colour_stim.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(colour_stim, 'tStartRefresh')  # time at next scr refresh
            colour_stim.setAutoDraw(True)
        if colour_stim.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > colour_stim.tStartRefresh + 0-frameTolerance:
                # keep track of stop time/frame for later
                colour_stim.tStop = t  # not accounting for scr refresh
                colour_stim.frameNStop = frameN  # exact frame index
                win.timeOnFlip(colour_stim, 'tStopRefresh')  # time at next scr refresh
                colour_stim.setAutoDraw(False)
        if colour_stim.status == STARTED:  # only update if drawing
            colour_stim.setFillColor(cir_color, log=False)
            colour_stim.setPos(cir_pos, log=False)
            colour_stim.setLineColor('white', log=False)
        
        # *probe_r_fill* updates
        if probe_r_fill.status == NOT_STARTED and tThisFlip >= probe_start-frameTolerance:
            # keep track of start time/frame for later
            probe_r_fill.frameNStart = frameN  # exact frame index
            probe_r_fill.tStart = t  # local t and not account for scr refresh
            probe_r_fill.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(probe_r_fill, 'tStartRefresh')  # time at next scr refresh
            probe_r_fill.setAutoDraw(True)
        if probe_r_fill.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > probe_r_fill.tStartRefresh + 0.1-frameTolerance:
                # keep track of stop time/frame for later
                probe_r_fill.tStop = t  # not accounting for scr refresh
                probe_r_fill.frameNStop = frameN  # exact frame index
                win.timeOnFlip(probe_r_fill, 'tStopRefresh')  # time at next scr refresh
                probe_r_fill.setAutoDraw(False)
        if probe_r_fill.status == STARTED:  # only update if drawing
            probe_r_fill.setFillColor(r_fill_color, log=False)
            probe_r_fill.setLineColor(r_fill_color, log=False)
        
        # *probe_l_fill* updates
        if probe_l_fill.status == NOT_STARTED and tThisFlip >= probe_start-frameTolerance:
            # keep track of start time/frame for later
            probe_l_fill.frameNStart = frameN  # exact frame index
            probe_l_fill.tStart = t  # local t and not account for scr refresh
            probe_l_fill.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(probe_l_fill, 'tStartRefresh')  # time at next scr refresh
            probe_l_fill.setAutoDraw(True)
        if probe_l_fill.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > probe_l_fill.tStartRefresh + 0.1-frameTolerance:
                # keep track of stop time/frame for later
                probe_l_fill.tStop = t  # not accounting for scr refresh
                probe_l_fill.frameNStop = frameN  # exact frame index
                win.timeOnFlip(probe_l_fill, 'tStopRefresh')  # time at next scr refresh
                probe_l_fill.setAutoDraw(False)
        if probe_l_fill.status == STARTED:  # only update if drawing
            probe_l_fill.setFillColor(l_fill_color, log=False)
            probe_l_fill.setLineColor(l_fill_color, log=False)
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        if text.status == STARTED:  # only update if drawing
            text.setText(aai_mean, log=False)
        
        # *key_resp* updates
        waitOnFlip = False
        if key_resp.status == NOT_STARTED and tThisFlip >= probe_start-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                key_resp.rt = [key.rt for key in _key_resp_allKeys]
                # a response ends the routine
                continueRoutine = False
        
        # *text2* updates
        if text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text2.frameNStart = frameN  # exact frame index
            text2.tStart = t  # local t and not account for scr refresh
            text2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text2, 'tStartRefresh')  # time at next scr refresh
            text2.setAutoDraw(True)
        if text2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text2.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text2.tStop = t  # not accounting for scr refresh
                text2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text2, 'tStopRefresh')  # time at next scr refresh
                text2.setAutoDraw(False)
        if text2.status == STARTED:  # only update if drawing
            text2.setText(aai_chunk_scaled, log=False)
        
        # *key_log* updates
        waitOnFlip = False
        if key_log.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_log.frameNStart = frameN  # exact frame index
            key_log.tStart = t  # local t and not account for scr refresh
            key_log.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_log, 'tStartRefresh')  # time at next scr refresh
            key_log.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_log.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_log.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_log.status == STARTED and not waitOnFlip:
            theseKeys = key_log.getKeys(keyList=['space'], waitRelease=False)
            _key_log_allKeys.extend(theseKeys)
            if len(_key_log_allKeys):
                key_log.keys = [key.name for key in _key_log_allKeys]  # storing all keys
                key_log.rt = [key.rt for key in _key_log_allKeys]
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('fixation_cross.started', fixation_cross.tStartRefresh)
    trials.addData('fixation_cross.stopped', fixation_cross.tStopRefresh)
    trials.addData('probe_l.started', probe_l.tStartRefresh)
    trials.addData('probe_l.stopped', probe_l.tStopRefresh)
    trials.addData('probe_r.started', probe_r.tStartRefresh)
    trials.addData('probe_r.stopped', probe_r.tStopRefresh)
    trials.addData('colour_stim.started', colour_stim.tStartRefresh)
    trials.addData('colour_stim.stopped', colour_stim.tStopRefresh)
    trials.addData('probe_r_fill.started', probe_r_fill.tStartRefresh)
    trials.addData('probe_r_fill.stopped', probe_r_fill.tStopRefresh)
    trials.addData('probe_l_fill.started', probe_l_fill.tStartRefresh)
    trials.addData('probe_l_fill.stopped', probe_l_fill.tStopRefresh)
    trials.addData('text.started', text.tStartRefresh)
    trials.addData('text.stopped', text.tStopRefresh)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    trials.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        trials.addData('key_resp.rt', key_resp.rt)
    trials.addData('key_resp.started', key_resp.tStartRefresh)
    trials.addData('key_resp.stopped', key_resp.tStopRefresh)
    trials.addData('text2.started', text2.tStartRefresh)
    trials.addData('text2.stopped', text2.tStopRefresh)
    # check responses
    if key_log.keys in ['', [], None]:  # No response was made
        key_log.keys = None
    trials.addData('key_log.keys',key_log.keys)
    if key_log.keys != None:  # we had a response
        trials.addData('key_log.rt', key_log.rt)
    trials.addData('key_log.started', key_log.tStartRefresh)
    trials.addData('key_log.stopped', key_log.tStopRefresh)
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 10.0 repeats of 'trials'


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
