#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.1),
    on August 23, 2022, at 15:13
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, parallel, hardware
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

# Run 'Before Experiment' code from code_2
from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import random 
import pylink
import time
from string import ascii_letters, digits

from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

#-------EYELINK PARAMS ---------------------------------------------------------
# from C:\Program Files (x86)\SR Research\EyeLink\SampleExperiments\Python\examples\Psychopy_examples\picture\picture.py
def show_msg(win, text, wait_for_keypress=True):
    """ Show task instructions on screen"""

    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)
        
def clear_screen(win):
    """ clear up the PsychoPy window"""

    win.fillColor = genv.getBackgroundColor()
    win.flip()
    
def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, wait_for_keypress=False)
        print(msg)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        print(f'EDF FILE PATH: {local_edf}')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()   
    
def abort_trial():
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR   
    


# get the time the probe appears
probe_start = random.uniform(4, 5.5)

# read in the test signal for the colour changes
cir_color = "blue"
block_text = 0

info = StreamInfo('PosnerMarkers', 'Markers', 1, 0, 'float32', 'posner_marker')

outlet = StreamOutlet(info)
outlet.push_sample([99])
#streams = resolve_stream('name', 'OutStream')

def rescale(sample, un_scaled_min=0, un_scaled_max=1, scaled_min=255, scaled_max=0):
    un_scaled_range = (un_scaled_max - un_scaled_min)
    scaled_range = (scaled_max - scaled_min)
    scaled_sample =(((sample-un_scaled_min) * scaled_range)/un_scaled_range) + scaled_min
    scaled_sample = np.clip(scaled_sample, 0, 255)
    return scaled_sample

  
    
#--DEFINE THE TRIALS FUNCTION---------------------------------------------------
def run_trials(thisTrial, trials, block=0):
    # ------------- MORE EYE TRACKER STUFF -----------------------------------------
    print('EYETRACKER TRIAL SETUP')
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()

    # clear the host screen before we draw the backdrop
    el_tracker.sendCommand('clear_screen 0')
    # OPTIONAL: draw landmarks and texts on the Host screen
    # In addition to backdrop image, You may draw simples on the Host PC to use
    # as landmarks. For illustration purpose, here we draw some texts and a box
    # For a list of supported draw commands, see the "COMMANDS.INI" file on the
    # Host PC (under /elcl/exe)
    left = int(scn_width/2.0) - 60
    top = int(scn_height/2.0) - 60
    right = int(scn_width/2.0) + 60
    bottom = int(scn_height/2.0) + 60
    draw_cmd = 'draw_filled_box %d %d %d %d 1' % (left, top, right, bottom)
    el_tracker.sendCommand(draw_cmd)
    # ==============================================================================  
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))

    for trial_index, thisTrial in enumerate(trials):
        print(f'STARTING TRIAL: {trials.thisN}')
        print(f'TRIALID {block}-{trial_index}')
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
            # send a "TRIALID" message to mark the start of a trial, see Data
            
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        el_tracker.sendMessage(f'TRIALID {block}-{trial_index}')  
        
        # record_status_message : show some info on the Host PC
        # here we show how many trial has been tested
        status_msg = f'TRIAL number {block}-{trial_index}'
        el_tracker.sendCommand("record_status_message '%s'" % status_msg)
        
        # put tracker in idle/offline mode before recording
        el_tracker.setOfflineMode()

        # Start recording
        # arguments: sample_to_file, events_to_file, sample_over_link,
        # event_over_link (1-yes, 0-no)
        try:
            el_tracker.startRecording(1, 1, 1, 1)
        except RuntimeError as error:
            print("ERROR:", error)
            abort_trial()
            return pylink.TRIAL_ERROR

        # Allocate some time for the tracker to cache some samples
        pylink.pumpDelay(100)
    
        # --- Prepare to start Routine "cue" ---
        continueRoutine = True
        # update component parameters for each repeat
        probe_r_cue.setPos((5, -1))
        # Run 'Begin Routine' code from code_2
        import random 
        
        probe_start = random.uniform(4, 5.5)
        side = random.choice([1,2,3]) # 1=l, 2=r, 3=n
        
        el_tracker.sendMessage(f'CUE_DIR-{side}')
        neutral_side = random.choice([1,2])
        
        # push the start of the cue
        #outlet.push_sample([f'cue_{side}'])
        outlet.push_sample([side])
        
        
        probe_location = (random.choice([-5, 5]), 0)
        # reset 'positions' 
        positions = copy.deepcopy(master_positions) 
        
        #randomise this for each trial
        random.shuffle(positions)
        
        
        if side == 1:
            r_cue_color = (0,0,0,0)
            l_cue_color = (255,255,255)
            c_cue_color = (0,0,0,0)
        elif side == 2:
            r_cue_color = (255,255,255)
            l_cue_color = (0,0,0,0)
            c_cue_color = (0,0,0,0)
        elif side == 3:
            r_cue_color = (0,0,0,0)
            l_cue_color = (0,0,0,0)
            c_cue_color = (255,255,255)
            
        valid_cue_weight = 70
        stim_side = random.choices([10, 11], weights=(valid_cue_weight, 100-valid_cue_weight))[0]
        if stim_side == 10:
            # valid cue
            if side == 1:
                r_fill_color = (0,0,0,0)
                l_fill_color = (255,255,255)
            elif side == 2:
                r_fill_color = (255,255,255)
                l_fill_color = (0,0,0,0)
            elif side == 3:
                # 50% chance left or right
                if neutral_side == 1:
                    r_fill_color = (0,0,0,0)
                    l_fill_color = (255,255,255)
                elif neutral_side == 2:
                    r_fill_color = (255,255,255)
                    l_fill_color = (0,0,0,0)    
        elif stim_side == 11:
            # invalid cue
            if side == 1:
                l_fill_color = (0,0,0,0)
                r_fill_color = (255,255,255)
            elif side == 2:
                l_fill_color = (255,255,255)
                r_fill_color = (0,0,0,0)
            elif side == 3:
                # 50% chance left or right
                if neutral_side == 1:
                    r_fill_color = (0,0,0,0)
                    l_fill_color = (255,255,255)
                elif neutral_side == 2:
                    r_fill_color = (255,255,255)
                    l_fill_color = (0,0,0,0)    
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        key_log.keys = []
        key_log.rt = []
        _key_log_allKeys = []
        # keep track of which components have finished
        cueComponents = [probe_l_cue, probe_r_cue, fc, left_cue, right_cue, centre_cue1, centre_cue2, probe_l_fill, probe_r_fill, p_port_cue, p_port_trial, key_resp, key_log]
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
        frameN = -1
        
        # --- Run Routine "cue" ---
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *probe_l_cue* updates
            if probe_l_cue.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                probe_l_cue.frameNStart = frameN  # exact frame index
                probe_l_cue.tStart = t  # local t and not account for scr refresh
                probe_l_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_l_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_l_cue.started')
                probe_l_cue.setAutoDraw(True)
            if probe_l_cue.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    probe_l_cue.tStop = t  # not accounting for scr refresh
                    probe_l_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_l_cue.stopped')
                    probe_l_cue.setAutoDraw(False)
            if probe_l_cue.status == STARTED:  # only update if drawing
                probe_l_cue.setFillColor('blue', log=False)
                probe_l_cue.setPos((-5, -1), log=False)
                probe_l_cue.setLineColor('white', log=False)
            
            # *probe_r_cue* updates
            if probe_r_cue.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                probe_r_cue.frameNStart = frameN  # exact frame index
                probe_r_cue.tStart = t  # local t and not account for scr refresh
                probe_r_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_r_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_r_cue.started')
                probe_r_cue.setAutoDraw(True)
            if probe_r_cue.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    probe_r_cue.tStop = t  # not accounting for scr refresh
                    probe_r_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_r_cue.stopped')
                    probe_r_cue.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
            
            import numpy as np
            
            
            r = random.randrange(0, 255)
            #g = random.randrange(0, 255)
            #b = random.randrange(0, 255)
            #print(f"colour = {g}")
            
            #r = 127* np.sin(t*2*pi)+127
            r = np.sin(t*2*pi)
            #r = 100+50*sin(t)**4
    #        print(f"{r} :- {frameN}: {cir_color}")
            cir_pos = ( sin(t*2*pi), cos(t*2*pi) )
            y_pos = 5* np.sin(t)
            cir_color = (r,0,0)
            
            
            
            # *fc* updates
            if fc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fc.frameNStart = frameN  # exact frame index
                fc.tStart = t  # local t and not account for scr refresh
                fc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fc.started')
                fc.setAutoDraw(True)
                el_tracker.sendMessage(f'FC_START')
            if fc.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fc.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fc.tStop = t  # not accounting for scr refresh
                    fc.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fc.stopped')
                    fc.setAutoDraw(False)
                    el_tracker.sendMessage(f'FC_STOP')
            
            # *left_cue* updates
            if left_cue.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                left_cue.frameNStart = frameN  # exact frame index
                left_cue.tStart = t  # local t and not account for scr refresh
                left_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue.started')
                left_cue.setAutoDraw(True)
                if l_cue_color ==(255,255,255):
                    el_tracker.sendMessage(f'LEFT_CUE_START')
            if left_cue.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    left_cue.tStop = t  # not accounting for scr refresh
                    left_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_cue.stopped')
                    left_cue.setAutoDraw(False)
                    if l_cue_color ==(255,255,255):
                        el_tracker.sendMessage(f'LEFT_CUE_STOP')
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue.started')
                right_cue.setAutoDraw(True)
                if r_cue_color ==(255,255,255):
                    el_tracker.sendMessage(f'RIGHT_CUE_START')
            if right_cue.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    right_cue.tStop = t  # not accounting for scr refresh
                    right_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_cue.stopped')
                    right_cue.setAutoDraw(False)
                    if r_cue_color ==(255,255,255):
                        el_tracker.sendMessage(f'RIGHT_CUE_STOP')
            if right_cue.status == STARTED:  # only update if drawing
                right_cue.setFillColor(r_cue_color, log=False)
                right_cue.setLineColor(r_cue_color, log=False)
            
            # *centre_cue1* updates
            if centre_cue1.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                centre_cue1.frameNStart = frameN  # exact frame index
                centre_cue1.tStart = t  # local t and not account for scr refresh
                centre_cue1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(centre_cue1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'centre_cue1.started')
                centre_cue1.setAutoDraw(True)
                if c_cue_color ==(255,255,255):
                    el_tracker.sendMessage(f'CENTRE_CUE_START')
            if centre_cue1.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    centre_cue1.tStop = t  # not accounting for scr refresh
                    centre_cue1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'centre_cue1.stopped')
                    centre_cue1.setAutoDraw(False)
                    if c_cue_color ==(255,255,255):
                        el_tracker.sendMessage(f'CENTRE_CUE_STOP')
            if centre_cue1.status == STARTED:  # only update if drawing
                centre_cue1.setFillColor(c_cue_color, log=False)
                centre_cue1.setLineColor(c_cue_color, log=False)
            
            # *centre_cue2* updates
            if centre_cue2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                centre_cue2.frameNStart = frameN  # exact frame index
                centre_cue2.tStart = t  # local t and not account for scr refresh
                centre_cue2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(centre_cue2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'centre_cue2.started')
                centre_cue2.setAutoDraw(True)
            if centre_cue2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    centre_cue2.tStop = t  # not accounting for scr refresh
                    centre_cue2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'centre_cue2.stopped')
                    centre_cue2.setAutoDraw(False)
            if centre_cue2.status == STARTED:  # only update if drawing
                centre_cue2.setFillColor(c_cue_color, log=False)
                centre_cue2.setLineColor(c_cue_color, log=False)
            
            # *probe_l_fill* updates
            if probe_l_fill.status == NOT_STARTED and tThisFlip >= probe_start-frameTolerance:
                # keep track of start time/frame for later
                probe_l_fill.frameNStart = frameN  # exact frame index
                probe_l_fill.tStart = t  # local t and not account for scr refresh
                probe_l_fill.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_l_fill, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_l_fill.started')
                probe_l_fill.setAutoDraw(True)
                if l_fill_color == (255,255,255):
                    el_tracker.sendMessage(f'LEFT_STIM_START')
            if probe_l_fill.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > probe_l_fill.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    probe_l_fill.tStop = t  # not accounting for scr refresh
                    probe_l_fill.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_l_fill.stopped')
                    probe_l_fill.setAutoDraw(False)
                    if l_fill_color == (255,255,255):
                        el_tracker.sendMessage(f'LEFT_STIM_STOP')
            if probe_l_fill.status == STARTED:  # only update if drawing
                probe_l_fill.setFillColor(l_fill_color, log=False)
                probe_l_fill.setLineColor(l_fill_color, log=False)
            
            # *probe_r_fill* updates
            if probe_r_fill.status == NOT_STARTED and tThisFlip >= probe_start-frameTolerance:
                # keep track of start time/frame for later
                probe_r_fill.frameNStart = frameN  # exact frame index
                probe_r_fill.tStart = t  # local t and not account for scr refresh
                probe_r_fill.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_r_fill, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_r_fill.started')
                probe_r_fill.setAutoDraw(True)
                if r_fill_color == (255,255,255):
                    el_tracker.sendMessage(f'RIGHT_STIM_START')
            if probe_r_fill.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > probe_r_fill.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    probe_r_fill.tStop = t  # not accounting for scr refresh
                    probe_r_fill.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_r_fill.stopped')
                    probe_r_fill.setAutoDraw(False)
                    if r_fill_color == (255,255,255):
                        el_tracker.sendMessage(f'RIGHT_STIM_STOP')
            if probe_r_fill.status == STARTED:  # only update if drawing
                probe_r_fill.setFillColor(r_fill_color, log=False)
                probe_r_fill.setLineColor(r_fill_color, log=False)
            # *p_port_cue* updates
            if p_port_cue.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                p_port_cue.frameNStart = frameN  # exact frame index
                p_port_cue.tStart = t  # local t and not account for scr refresh
                p_port_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port_cue.started', t)
                p_port_cue.status = STARTED
                win.callOnFlip(p_port_cue.setData, int(side))
            if p_port_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port_cue.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port_cue.tStop = t  # not accounting for scr refresh
                    p_port_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port_cue.stopped', t)
                    p_port_cue.status = FINISHED
                    win.callOnFlip(p_port_cue.setData, int(0))
            # *p_port_trial* updates
            if p_port_trial.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                p_port_trial.frameNStart = frameN  # exact frame index
                p_port_trial.tStart = t  # local t and not account for scr refresh
                p_port_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port_trial.started', t)
                p_port_trial.status = STARTED
                win.callOnFlip(p_port_trial.setData, int(side))
            if p_port_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port_trial.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port_trial.tStop = t  # not accounting for scr refresh
                    p_port_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port_trial.stopped', t)
                    p_port_trial.status = FINISHED
                    win.callOnFlip(p_port_trial.setData, int(0))
            
            # *key_resp* updates
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= probe_start-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['right', 'left'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                    key_resp.rt = [key.rt for key in _key_resp_allKeys]
            
            # *key_log* updates
            waitOnFlip = False
            if key_log.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_log.frameNStart = frameN  # exact frame index
                key_log.tStart = t  # local t and not account for scr refresh
                key_log.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_log, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_log.started')
                key_log.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_log.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_log.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_log.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6.6-frameTolerance:
                    # keep track of stop time/frame for later
                    key_log.tStop = t  # not accounting for scr refresh
                    key_log.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_log.stopped')
                    key_log.status = FINISHED
            if key_log.status == STARTED and not waitOnFlip:
                theseKeys = key_log.getKeys(keyList=['right', 'left'], waitRelease=False)
                _key_log_allKeys.extend(theseKeys)
                if len(_key_log_allKeys):
                    key_log.keys = [key.name for key in _key_log_allKeys]  # storing all keys
                    key_log.rt = [key.rt for key in _key_log_allKeys]
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                terminate_task()
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
        
        # --- Ending Routine "cue" ---
        for thisComponent in cueComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from code_2
        thisExp.addData('cue', side)
        thisExp.addData('stim_side', stim_side)
        probe_start = 0
        if p_port_cue.status == STARTED:
            win.callOnFlip(p_port_cue.setData, int(0))
        if p_port_trial.status == STARTED:
            win.callOnFlip(p_port_trial.setData, int(0))
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
        # check responses
        if key_log.keys in ['', [], None]:  # No response was made
            key_log.keys = None
        trials.addData('key_log.keys',key_log.keys)
        if key_log.keys != None:  # we had a response
            trials.addData('key_log.rt', key_log.rt)
        # the Routine "cue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 25.0 repeats of 'trials'
    # clear the screen
    clear_screen(win)
    el_tracker.sendMessage('blank_screen')
    # send a message to clear the Data Viewer screen as well
    el_tracker.sendMessage('!V CLEAR 128 128 128')

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()
    # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)

def run_continue(block_no):
    # --- Prepare to start Routine "continue_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    continue_response.keys = []
    continue_response.rt = []
    _continue_response_allKeys = []
    # Run 'Begin Routine' code from code_4
#    block_no = block_no + 1

    block_text = f"block {block_no}/4 completed"
    text_2.setText(block_text
    )
    # keep track of which components have finished
    continue_2Components = [continue_instructions, continue_response, text_2]
    for thisComponent in continue_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "continue_2" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *continue_instructions* updates
        if continue_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_instructions.frameNStart = frameN  # exact frame index
            continue_instructions.tStart = t  # local t and not account for scr refresh
            continue_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_instructions.started')
            continue_instructions.setAutoDraw(True)
        
        # *continue_response* updates
        waitOnFlip = False
        if continue_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_response.frameNStart = frameN  # exact frame index
            continue_response.tStart = t  # local t and not account for scr refresh
            continue_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_response, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_response.started')
            continue_response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_response.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_response.status == STARTED and not waitOnFlip:
            theseKeys = continue_response.getKeys(keyList=['return', 'down'], waitRelease=False)
            _continue_response_allKeys.extend(theseKeys)
            if len(_continue_response_allKeys):
                continue_response.keys = _continue_response_allKeys[-1].name  # just the last key pressed
                continue_response.rt = _continue_response_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            text_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            terminate_task()
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in continue_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "continue_2" ---
    for thisComponent in continue_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if continue_response.keys in ['', [], None]:  # No response was made
        continue_response.keys = None
    thisExp.addData('continue_response.keys',continue_response.keys)
    if continue_response.keys != None:  # we had a response
        thisExp.addData('continue_response.rt', continue_response.rt)
    thisExp.nextEntry()
    # the Routine "continue_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
#===============================================================================

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.1'
expName = 'posner_task'  # from the Builder filename that created this script
expInfo = {
    'participant': '',
    'session': '001',
}

# --- Show participant info dialog --
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
    originPath='C:\\Users\\2354158T\\Documents\\GitHub\\nfb\\psychopy\\posner_eyelink.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1280, 1024], fullscr=True, screen=1, 
    winType='pyglet', allowStencil=False,
    monitor='eprime', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='deg')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
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


# =============DO THE EYE TRACKER SETUP AND CALIBRATION
#...............................................................................    
dummy_mode = False
edf_fname = f"{expInfo['participant']}_psnr"
results_folder = 'results'
eyelink_ip = "100.1.1.1"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)
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
        el_tracker = pylink.EyeLink(eyelink_ip)
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()
        
# Step 2: Open an EDF data file on the Host PC
edf_file = edf_fname + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()
scn_width, scn_height = win.size
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
else:
    print(f'Running experiment in dummy mode')
    
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

# calibration type
el_tracker.sendCommand("calibration_type = HV5")
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
#el_tracker.sendCommand("button_function 5 'accept_target_fixation'")


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
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
print(genv)  # print out the version number of the CoreGraphics library

# Set background and foreground colors for the calibration target
# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
foreground_color = (-1, -1, -1)
background_color = win.color
genv.setCalibrationColors(foreground_color, background_color)

# Set up the calibration target
#
# The target could be a "circle" (default), a "picture", a "movie" clip,
# or a rotating "spiral". To configure the type of calibration target, set
# genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
# genv.setTargetType('picture')
# Use a picture as the calibration target
genv.setTargetType('circle')

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle" and "spiral" targets
# genv.setTargetSize(24)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('off', 'off', 'off')

# Request Pylink to use the PsychoPy window we opened above for calibration
print(f"STARTING CALIBRARION")
pylink.openGraphicsEx(genv)

# Step 5: Set up the camera and calibrate the tracker

## Show the task instructions
#task_msg = 'In the task, you may press the SPACEBAR to end a trial\n' + \
#    '\nPress Ctrl-C to if you need to quit the task early\n'
#if dummy_mode:
#    task_msg = task_msg + '\nNow, press ENTER to start the task'
#else:
#    task_msg = task_msg + '\nNow, press ENTER twice to calibrate tracker'
#show_msg(win, task_msg)

# skip this step if running the script in Dummy Mode
if not dummy_mode:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()
print(f"CALIBRARION DONE")
        
#win.close()
#===============================================================================
# =====================================================

# --- Initialize components for Routine "start" ---
start_instructions = visual.TextStim(win=win, name='start_instructions',
    text='This is the start of the experiment.\npress the down arrow to start.',
    font='Open Sans',
    units='cm', pos=(0, 0), height=1.5, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
start_resp = keyboard.Keyboard()

# --- Initialize components for Routine "cue" ---
probe_l_cue = visual.ShapeStim(
    win=win, name='probe_l_cue',units='deg', 
    size=(3.5, 3.5), vertices='circle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=8.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
probe_r_cue = visual.ShapeStim(
    win=win, name='probe_r_cue',units='deg', 
    size=(3.5, 3.5), vertices='circle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=8.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
    opacity=None, depth=-1.0, interpolate=True)
# Run 'Begin Experiment' code from code_2
import random, copy
from random import randint
from decimal import *

# setup an array to store 6 locations to randomly allocate to our shapes 
# (or select random locations between (-1,-1) and (1,1) if you want them completely 
# random though you then need to factor in overlaying shapes)
master_positions=[[-5,0], [5,0]]
positions = copy.deepcopy(master_positions)
fc = visual.ShapeStim(
    win=win, name='fc',units='cm', 
    size=(0.25, 0.25), vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)
left_cue = visual.ShapeStim(
    win=win, name='left_cue',units='cm', 
    size=(0.75, 0.75), vertices='triangle',
    ori=-90.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-4.0, interpolate=True)
right_cue = visual.ShapeStim(
    win=win, name='right_cue',units='cm', 
    size=(0.75, 0.75), vertices='triangle',
    ori=90.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
centre_cue1 = visual.ShapeStim(
    win=win, name='centre_cue1',units='cm', 
    size=(0.75, 0.75), vertices='triangle',
    ori=90.0, pos=(0.375, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-6.0, interpolate=True)
centre_cue2 = visual.ShapeStim(
    win=win, name='centre_cue2',units='cm', 
    size=(0.75, 0.75), vertices='triangle',
    ori=-90.0, pos=(-0.375, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-7.0, interpolate=True)
probe_l_fill = visual.ShapeStim(
    win=win, name='probe_l_fill',units='deg', 
    size=(1, 1), vertices='circle',
    ori=0.0, pos=(-5,-1), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-8.0, interpolate=True)
probe_r_fill = visual.ShapeStim(
    win=win, name='probe_r_fill',units='deg', 
    size=(1, 1), vertices='circle',
    ori=0.0, pos=(5,-1), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-9.0, interpolate=False)
p_port_cue = parallel.ParallelPort(address='0x2010')
p_port_trial = parallel.ParallelPort(address='0x2010')
key_resp = keyboard.Keyboard()
key_log = keyboard.Keyboard()

# --- Initialize components for Routine "continue_2" ---
continue_instructions = visual.TextStim(win=win, name='continue_instructions',
    text='have a break. Press the down arrow to continue when ready.',
    font='Open Sans',
    units='cm', pos=(0, 0), height=1.5, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
continue_response = keyboard.Keyboard()
text_2 = visual.TextStim(win=win, name='text_2',
    text='',
    font='Open Sans',
    units='cm', pos=(0, -4.5), height=1.5, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# --- Initialize components for Routine "end" ---
end_text = visual.TextStim(win=win, name='end_text',
    text='Task finished, wait for experimenter.',
    font='Open Sans',
    units='cm', pos=(0, 0), height=2.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
end_key = keyboard.Keyboard()
p_port_3 = parallel.ParallelPort(address='0x2010')

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "start" ---
continueRoutine = True
block_no = 0
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
frameN = -1

# --- Run Routine "start" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
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
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'start_instructions.started')
        start_instructions.setAutoDraw(True)
    
    # *start_resp* updates
    waitOnFlip = False
    if start_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        start_resp.frameNStart = frameN  # exact frame index
        start_resp.tStart = t  # local t and not account for scr refresh
        start_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(start_resp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'start_resp.started')
        start_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(start_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(start_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if start_resp.status == STARTED and not waitOnFlip:
        theseKeys = start_resp.getKeys(keyList=['return', 'v', 'b', 'down'], waitRelease=False)
        _start_resp_allKeys.extend(theseKeys)
        if len(_start_resp_allKeys):
            start_resp.keys = _start_resp_allKeys[-1].name  # just the last key pressed
            start_resp.rt = _start_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        terminate_task()
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

# --- Ending Routine "start" ---
for thisComponent in startComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# Run 'End Routine' code from code_3
outlet.push_sample([88])
# check responses
if start_resp.keys in ['', [], None]:  # No response was made
    start_resp.keys = None
thisExp.addData('start_resp.keys',start_resp.keys)
if start_resp.keys != None:  # we had a response
    thisExp.addData('start_resp.rt', start_resp.rt)
thisExp.nextEntry()
# the Routine "start" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials_ = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials_)  # add the loop to the experiment
thisTrial_ = trials_.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)

run_trials(thisTrial=thisTrial_, trials=trials_, block=block_no)
block_no = block_no + 1
run_continue(block_no)

# set up handler to look after randomisation of conditions etc
trials_2 = data.TrialHandler(nReps=2.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials_2')
thisExp.addLoop(trials_2)  # add the loop to the experiment
thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values

run_trials(thisTrial=thisTrial_2, trials=trials_2, block=block_no)
block_no = block_no + 1
run_continue(block_no)

# set up handler to look after randomisation of conditions etc
trials_3 = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials_3')
thisExp.addLoop(trials_3)  # add the loop to the experiment
thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values

run_trials(thisTrial=thisTrial_3, trials=trials_3, block=block_no)
block_no = block_no + 1
run_continue(block_no)

# set up handler to look after randomisation of conditions etc
trials_4 = data.TrialHandler(nReps=2.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials_4')
thisExp.addLoop(trials_4)  # add the loop to the experiment
thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values

run_trials(thisTrial=thisTrial_4, trials=trials_4, block=block_no)

# --- Prepare to start Routine "end" ---
continueRoutine = True
# update component parameters for each repeat
end_key.keys = []
end_key.rt = []
_end_key_allKeys = []
# keep track of which components have finished
endComponents = [end_text, end_key, p_port_3]
for thisComponent in endComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "end" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *end_text* updates
    if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_text.frameNStart = frameN  # exact frame index
        end_text.tStart = t  # local t and not account for scr refresh
        end_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'end_text.started')
        end_text.setAutoDraw(True)
    
    # *end_key* updates
    waitOnFlip = False
    if end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_key.frameNStart = frameN  # exact frame index
        end_key.tStart = t  # local t and not account for scr refresh
        end_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_key, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'end_key.started')
        end_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(end_key.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if end_key.status == STARTED and not waitOnFlip:
        theseKeys = end_key.getKeys(keyList=['space'], waitRelease=False)
        _end_key_allKeys.extend(theseKeys)
        if len(_end_key_allKeys):
            end_key.keys = _end_key_allKeys[-1].name  # just the last key pressed
            end_key.rt = _end_key_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    # *p_port_3* updates
    if p_port_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        p_port_3.frameNStart = frameN  # exact frame index
        p_port_3.tStart = t  # local t and not account for scr refresh
        p_port_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(p_port_3, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.addData('p_port_3.started', t)
        p_port_3.status = STARTED
        win.callOnFlip(p_port_3.setData, int(75))
    if p_port_3.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > p_port_3.tStartRefresh + 0.1-frameTolerance:
            # keep track of stop time/frame for later
            p_port_3.tStop = t  # not accounting for scr refresh
            p_port_3.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.addData('p_port_3.stopped', t)
            p_port_3.status = FINISHED
            win.callOnFlip(p_port_3.setData, int(0))
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        terminate_task()
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "end" ---
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if end_key.keys in ['', [], None]:  # No response was made
    end_key.keys = None
thisExp.addData('end_key.keys',end_key.keys)
if end_key.keys != None:  # we had a response
    thisExp.addData('end_key.rt', end_key.rt)
thisExp.nextEntry()
if p_port_3.status == STARTED:
    win.callOnFlip(p_port_3.setData, int(0))
# the Routine "end" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- End experiment ---
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

# Terminate the eye tracking stuff
terminate_task()

#Close up
win.close()
core.quit()
