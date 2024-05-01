#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import random

import numpy

from constants import *

import pygaze
from pygaze.display import Display
from pygaze.keyboard import Keyboard
from pygaze.eyetracker import EyeTracker
from pygaze.screen import Screen
from pygaze.sound import Sound
import pygaze.libtime as timer


# # # # #
# INITIALE

# Create a new display to make the monitor do things with.
disp = Display()

# Create a holding screen.
scr = Screen()
scr.draw_text("Loading, please wait...", fontsize=42)
disp.fill(scr)
disp.show()

# Create a Keyboard instance.
kb = Keyboard()

# Create a new EyeTracker instance.
tracker = EyeTracker(disp)

### Create a bunch of stimulus screens for our experiments. ###
scr_blank = Screen()
#fixation screen
scr_fix = Screen()
scr_fix.draw_fixation(fixtype="cross", diameter = 12)

# Calibration screen has 12 dots on it.
scr_calib = {}
for (px, py) in CALIB_DOTS_PX:
    _scr = Screen()
    _scr.draw_fixation(fixtype="dot", pos=(px, py), diameter=12)
    scr_calib[(px,py)] = _scr

# Stimulus screen will be used to draw images on. It's cleared every trial, and
# prepared at trial onset.

scr_img = {}
for imgname in IMGNAMES:
    imgpath = os.path.join(IMGDIR, imgname)
    _scr = Screen()
    _scr.draw_image(imgpath, scale=0.5)
    scr_img[imgname] = _scr

# Pupil experiment screens come in two types: one with just dark or just 
# bright, and one with dark and bright disks.
scr_dark = Screen()
scr_dark.clear(colour= COLOUR_DARK)
scr_dark.draw_fixation(fixtype="cross", diameter=12, colour=(125,125,125))

scr_light = Screen()
scr_light.clear(colour= COLOUR_LIGHT)
scr_light.draw_fixation(fixtype="cross", diameter=12, colour=(125,125,125))

# Draw disks on the screens.
scr_dark_left = Screen()
scr_dark_left.draw_fixation(fixtype="cross", diameter=12)
scr_dark_left.draw_circle(colour=COLOUR_DARK, pos=DISK_LEFT, r=DISK_RADIUS, fill=True)
scr_dark_left.draw_circle(colour=COLOUR_LIGHT, pos=DISK_RIGHT, r=DISK_RADIUS, fill=True)

scr_dark_right = Screen()
scr_dark_right.draw_fixation(fixtype="cross", diameter=12)
scr_dark_right.draw_circle(colour=COLOUR_DARK, pos=DISK_RIGHT, r=DISK_RADIUS, fill=True)
scr_dark_right.draw_circle(colour=COLOUR_LIGHT, pos=DISK_LEFT, r=DISK_RADIUS, fill=True)

#Draw target screens
scr_target = {}
for dark_left in [1,0]:
    scr_target[dark_left] = {}
    for target_left in [1,0]:
        scr_target[dark_left][target_left] = {}
        for target_up in [1,0]:
            scr_target[dark_left][target_left][target_up] = Screen()
            if dark_left:
                scr_target[dark_left][target_left][target_up].copy( \
                    scr_dark_left)
            else:
                scr_target[dark_left][target_left][target_up].copy( \
                    scr_dark_right)
            if target_left:
                spos = (DISK_LEFT[0], DISK_LEFT[1]-DISK_RADIUS//4)
            else:
                spos = (DISK_RIGHT[0], DISK_RIGHT[1]-DISK_RADIUS//4)
            epos = (spos[0], spos[1]+DISK_RADIUS//2)
            scr_target[dark_left][target_left][target_up].draw_line( \
                colour=(127, 127, 127), spos=spos, epos=epos, pw=3)
            if target_up:
                spos = (spos[0]-DISK_RADIUS//4, spos[1])
                epos = (spos[0]+DISK_RADIUS//2, spos[1])
            else:
                spos = (spos[0]-DISK_RADIUS//4, epos[1])
                epos = (spos[0]+DISK_RADIUS//2, spos[1])
            scr_target[dark_left][target_left][target_up].draw_line( \
                colour=(127, 127, 127), spos=spos, epos=epos, pw=3)

# Initialise sounds.
sound_left = Sound(soundfile=(SOUNDLEFT))
sound_right = Sound(soundfile=(SOUNDRIGHT))
sound_beep = Sound(freq=400, length=150)
sound_bad_beep = Sound(freq=3000, length=1000)

###  Create a trial list.  ###
trials = []
# Add the calibration trials at the start of the experiment.
calib_trials = []
for i, (px, py) in enumerate(CALIB_DOTS):
    trial = {}
    trial["type"] = "calibration"
    trial["nr"] = i
    trial["pos"] = (px, py)
    trial["pos_px"] = CALIB_DOTS_PX[i]
    calib_trials.append(trial)
# Randomise the trials.
random.shuffle(calib_trials)
# Add the trials to the whole set.
trials.extend(calib_trials)

##Add the image trials to trial list
img_trials = []
for i, imgname in enumerate(IMGNAMES):
    trial = {}
    trial["type"] = "images"
    trial["nr"] = i
    trial["imgname"] = imgname
    trial["imgpath"] = os.path.join(IMGDIR, imgname)
    img_trials.append(trial)
#randomise trials
random.shuffle(img_trials)
# add trials to whole set
trials.extend(img_trials)

# Add the standard pupil trials
pupil_trials = []
for i in range(PUPILTRIALS):
    trial = {}
    trial["type"] = "pupil"
    trial["nr"] = i
    if i % 2 == 0:
        trial["stimulus"] = "dark"
    else:
        trial["stimulus"] = "light"
    pupil_trials.append(trial)
# Note: these trials are NOT randomised, because we're using their alternation
# to evoke a pupil response.
#add to list
trials.extend(pupil_trials)

# Add the attention pupil trials
att_pupil_trials = []
for i in range(PUPILTRIALS):
    trial = {}
    trial["type"] = "att_pupil"
    trial["nr"] = i
    if numpy.random.rand() < 0.5:
        trial["stimulus"] = "dark_left"
    else:
        trial["stimulus"] = "dark_right"
    trial["dark_left"] = trial["stimulus"] == "dark_left"
    if numpy.random.rand() < 0.5:
        trial["attend_left"] = 1
    else:
        trial["attend_left"] = 0
    if ((trial["attend_left"]==1) and (trial["stimulus"] == "dark_left")) or \
        ((trial["attend_left"]==0) and (trial["stimulus"] == "dark_right")):
        trial["attend_dark"] = 1
    else:
        trial["attend_dark"] = 0
    if numpy.random.rand() < P_VALIDITY:
        trial["valid"] = 1
    else:
        trial["valid"] = 0
    if (trial["valid"] and trial["attend_left"]) or \
        ((trial["valid"]==0) and (trial["attend_left"]==0)):
        trial["target_left"] = 1
    else:
        trial["target_left"] = 0
    if numpy.random.rand() < 0.5:
        trial["target_up"] = 1
    else:
        trial["target_up"] = 0
    att_pupil_trials.append(trial)
#randomise trials
random.shuffle(att_pupil_trials)
#add to list
trials.extend(att_pupil_trials)

print(trials)

print("\n\n")
# Copy the first batch of calibration trials.
calib_post_trials = copy.deepcopy(calib_trials)
# Rename the trial types.
for ct in calib_post_trials:
    ct["type"] = "calibration_post"
# Randomise the calibration trials so they are in a different pseudo-random
# order than the first round of calibration trials.
random.shuffle(calib_post_trials)
# Add the trials to the whole set.
trials.extend(calib_post_trials)


# # # # # 
# RUN

#run calibration
tracker.calibrate()


previous_trial_type = None
for trialnr, trial in enumerate(trials):
    
    # Check if we need to run an instruction.
    if trial["type"] != previous_trial_type:
        # Write the instruction.
        scr.clear()
        scr.draw_text(text=INSTRUCTIONS[trial["type"]], fontsize=24)
        # Show the instruction.
        disp.fill(scr)
        disp.show()
        # Wait for a keyboard input.
        kb.get_key(keylist=None, timeout=None, flush=True)
    
    # # # # #
    # RUN THE TRIAL
    
    # Start recording eye movements.
    tracker.start_recording()
    
    # Run the specific trial type.
    if trial["type"] in ["calibration", "calibration_post"]:
        
        # Log trial start.
        tracker.log(("CALIBRATION_TRIALSTART, phase={}, trialnr={}, x={}, " \
            + "y={}").format(trial["type"], trial["nr"], trial["pos_px"][0], \
            trial["pos_px"][1]))
        
        # Briefly show a blank screen.
        disp.fill(scr_blank)
        disp.show()
        tracker.log("CALIB_BLANK_SCREEN")
        timer.pause(CALIB_BLANK_DUR)
        
        # Show the calibration dot.
        disp.fill(scr_calib[trial["pos_px"]])
        disp.show()
        sound_beep.play()
        tracker.log("CALIB_DOT_ONSET, x={}, y={}".format(trial["pos_px"][0], \
            trial["pos_px"][1]))
       
        # Wait until eyes have reacher the dot, and then pause.
        currentdis = numpy.inf
        t0 = timer.get_time()
        while currentdis >= THRESHOLD:
            
            #update the new eye position
            currenteyepos = tracker.sample()
            if currenteyepos == (-1,-1) or currenteyepos[0] == -1 or \
                currenteyepos[1] == -1 or currenteyepos[0] == None or \
                currenteyepos[1] == None or currenteyepos == (0,0):
                continue
            #update current distance
            currentdis = linedis(trial['pos_px'][0], currenteyepos[0],\
                                 trial['pos_px'][1], currenteyepos[1])
            # Break on a timeout.
            if timer.get_time() - t0 > 3000:
                break
        tracker.log("CALIB_FIXATION_DETECTED")
        timer.pause(CALIB_DOT_DUR)
        tracker.log("CALIB_TRIALSTOP")


    elif trial["type"] == "images":
        
        # Log trial start.
        tracker.log("IMAGE_TRIALSTART, imgname={}".format(trial["imgname"]))
        
        #fixation screen
        disp.fill(scr_fix)
        disp.show()
        tracker.log("IMAGE_FIXATION")
        timer.pause(IMG_FIXTIME)
        
        #show the image
        disp.fill(scr_img[trial["imgname"]])
        disp.show()
        tracker.log("IMAGE_ONSET")
        timer.pause(IMGTIME)
        
        # Redraw the fixation.
        disp.fill(scr_blank)
        disp.show()
        tracker.log("IMAGE_OFFSET")
        
        # Inter-trial interval.
        iti = random.randint(IMG_ITI[0], IMG_ITI[1])
        timer.pause(iti)

        # Log trial end.
        tracker.log("IMAGE_TRIALSTOP")
        

    elif trial["type"] == "pupil":
        
        #show the correct screen
        if trial["stimulus"] == "dark":
                disp.fill(scr_dark)
        else:
                disp.fill(scr_light)
        #present screen
        stim_onset = disp.show()
        
        # Record trial start.
        tracker.log("PUPIL_TRIALSTART, stimulus={}".format(trial["stimulus"]))
        #wait for pupil response 
        timer.pause(PUPTRIALTIME)
        # Log trial end.
        tracker.log("PUPIL_TRIALSTOP")

        #inter-trial interval
        iti = random.randint(PUP_ITI[0], PUP_ITI[1])
        timer.pause(iti)
        tracker.log("PUPIL_ITISTOP")
    
    
    elif trial["type"] == "att_pupil":
        
        # Present the basic screen.
        if trial["stimulus"] == "dark_left":
            disp.fill(scr_dark_left)
        else:
            disp.fill(scr_dark_right)
        # Show the display.
        stim_onset = disp.show()
        
        # Record trial start.
        tracker.log(("ATT_PUPIL_TRIALSTART, disk_position={},"\
            + " cued_position={}, target_location={}, target_orientation={},"\
            " target_congurent={}")\
            .format(trial["stimulus"], trial["attend_left"], \
                    trial["target_left"], trial["target_up"], trial["valid"]))

        # Wait for a bit.
        timer.pause(ATT_PUP_FIXTIME)

        # Present the cue.
        # Play a sound here.
        if trial["attend_left"]:
            sound_left.play()
        else:
            sound_right.play()
        # Log (rough) sound onset.
        tracker.log("ATT_PUPIL_CUE_SOUND_ONSET, cued location={}"\
            .format(trial["attend_left"]))
        
        # Wait for a bit.
        timer.pause(ATT_PUP_POSTCUETIME)
        
        #check if P is fixated
        currenteyepos = tracker.sample()
        centre = (DISPSIZE[0]*0.5,DISPSIZE[1]*0.5)
        currentdis = linedis(centre[0], currenteyepos[0],\
                             centre[1], currenteyepos[1])
        #t0 = timer.get_time()
        while currentdis >= BEEP_THRESHOLD:
            
            sound_bad_beep.play()
            #update the new eye position
            currenteyepos = tracker.sample()
            if currenteyepos == (-1,-1) or currenteyepos[0] == -1 or \
                currenteyepos[1] == -1 or currenteyepos[0] == None or \
                currenteyepos[1] == None or currenteyepos == (0,0):
                continue
            #update current distance
            currentdis = linedis(centre[0], currenteyepos[0],\
                                 centre[1], currenteyepos[1])
            # Break on a timeout.
            #if timer.get_time() - t0 > 3000:
                #break
        # Present the target.
        disp.fill(scr_target[trial["dark_left"]] \
            [trial["target_left"]][trial["target_up"]])
        target_onset = disp.show()
        
        tracker.log(("ATT_PUPIL_TARGET_ONSET, target_location={},"\
            + " target_orientation={}")\
            .format(trial["target_left"], trial["target_up"]))
        
        # Get the response.
        resp, click_time = kb.get_key(keylist=["up", "down"], timeout=None, \
            flush=True)
        resp_time = click_time - target_onset
        if trial["target_up"] and (resp == "up"):
            correct = 1
        elif (not trial["target_up"]) and (resp == "down"):
            correct = 1
        else:
            correct = 0
        
        # Log trial end.
        tracker.log("ATT_PUPIL_TRIALSTOP, response={}, correct={}"\
                    .format(resp, correct))
        
        # Show a blank screen during the inter-trial interval.
        disp.fill(scr_blank)
        disp.show()
        iti = random.randint(PUP_ITI[0], PUP_ITI[1])
        timer.pause(iti)

    # Stop recording eye movements.
    tracker.stop_recording()
            
    # Update the previous trial variable.
    previous_trial_type = copy.deepcopy(trial["type"]) 
     

# # # # #
# CLOSE

# Show a holding message.
scr.clear()
scr.draw_text("Shutting down, please wait...", fontsize=42)
disp.fill(scr)
disp.show()

# Close the connection to the tracker.
tracker.close()

# Show an ending message.
scr.clear()
scr.draw_text( \
    "That's all! Thanks for participating.\n\n(Press any key to exit.)", \
    fontsize=42)
disp.fill(scr)
disp.show()
kb.get_key(flush=True, keylist=None, timeout=None)

# Ends the display connection and the experiment.
disp.close()
