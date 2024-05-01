#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math

def deg2px(v, d, px_per_cm):
    s_cm = 2 * d * math.tan(math.radians(v)/2.0)
    return s_cm*px_per_cm

def px2deg(s, d, px_per_cm):
    v = math.degrees(2 * math.arctan((s/px_per_cm) / (2*d)))
    return v

def linedis(x1, x2, y1, y2):
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

# XEYE, YEYE, XDOT, YDOT

# Display and screen things.
# SCREENNR = 0
DISPTYPE = "pygame"
DISPSIZE = (1440,900) #(1920, 1080)
SCREENSIZE = (42.0, 26.0)
PX_PER_CM = ((DISPSIZE[0]/SCREENSIZE[0]) + (DISPSIZE[1]/SCREENSIZE[1])) / 2.0
SCREENDIST = 60.0
FGC = (0, 0, 0)
BGC = (127, 127, 127)

# Stimulus settings.
# The calibration screen has 12 dots, spread across 5 rows.
CALIB_DOTS = [ \
    (0.3, 0.1), (0.7, 0.1), \
    (0.1, 0.3), (0.5, 0.3), (0.9, 0.3), \
    (0.3, 0.5), (0.7, 0.5), \
    (0.1, 0.7), (0.5, 0.7), (0.9, 0.7), \
    (0.3, 0.9), (0.7, 0.9), \
    ]
CALIB_DOTS_PX = []
for (px, py) in CALIB_DOTS:
    CALIB_DOTS_PX.append((round(DISPSIZE[0]*px), round(DISPSIZE[1]*py)))

# Files and folders.
DIR = os.path.dirname(os.path.abspath('__file__'))
DATADIR = os.path.join(DIR, "data")
if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)

#create a list of images 
IMGDIR = os.path.join(DIR, 'images')
IMGNAMES = os.listdir(IMGDIR)
for fname in IMGNAMES:
    name, ext = os.path.splitext(fname)
    if ext not in [".png", ".jpg", ".jpeg"]:
        IMGNAMES.remove(fname)

#creates filepath for sounds 
SOUNDDIR = os.path.join(DIR, 'sound')
SOUNDLEFT = os.path.join(SOUNDDIR, 'left.wav')
SOUNDRIGHT = os.path.join(SOUNDDIR, 'right.wav')

# RGB values for dark and light screens.
COLOUR_DARK = (0, 0, 0)
COLOUR_LIGHT = (255, 255, 255)

#STDPUPIL snd ATTPUPIL trials
PUPILTRIALS = 50 #50

#Disk ATTPUPIL coordinates
# Proportion of trials that are validly cued.
P_VALIDITY = 0.8
# Set to eccentricity of 9.2 degrees of visual angle.
DISP_CENTRE = (DISPSIZE[0]//2, DISPSIZE[1]//2)
eccentricity = deg2px(9.2, SCREENDIST, PX_PER_CM)
DISK_LEFT = (DISP_CENTRE[0] - eccentricity, DISP_CENTRE[1])
DISK_RIGHT = (DISP_CENTRE[0] + eccentricity, DISP_CENTRE[1])
# Set to radius of 3.1 degrees of visual angle.
DISK_RADIUS = deg2px(3.1, SCREENDIST, PX_PER_CM)

THRESHOLD = deg2px(1.5, SCREENDIST, PX_PER_CM)

BEEP_THRESHOLD = deg2px(2.5, SCREENDIST, PX_PER_CM)

# Written instructions for each trial type.
INSTRUCTIONS = { \
    "calibration": "Please look at each point. Keep looking until the point " \
        + "disappears.", \

    "calibration_post": "Please look at each point. Keep looking until the " \
        + "point disappears.", \

    "images": "Please view the following images. You can look however you " \
        + "would like. There are no further instructions.", \
    
    "pupil": "You will see alternating screen brightnesses. Please keep " \
        + "looking at the central dot.", \
    
    "att_pupil": "Please keep your eyes on the central dot at all times. A" \
        + "letter 'T' will appear in the left or the right disk. When it " \
        + "does, please indicate whether it is:\n\n " \
        + "    - the right way up (arrow key UP), or\n" \
        + "    - upside down (arrow key DOWN)" \
        + "\n\nA voice will tell you in which disk the 'T' is most likely" \
        + " to appear." \
        + "\nA sound will play if you move your eyes off the cross."
    }
for trial_type in INSTRUCTIONS.keys():
    INSTRUCTIONS[trial_type] += "\n\n\n(Press any key to start.)"

# Get participant details.
LOGFILENAME = input("\nWhat is the file name? ")
LOGFILE = os.path.join(DATADIR, LOGFILENAME)

# Eye tracker things.
DUMMYMODE = False
TRACKERTYPE = "alea"

#Timings 
ITI = 1000
POINTTIME = 2000
CALIB_BLANK_DUR = 500
CALIB_DOT_DUR = 1500
IMG_FIXTIME = 1000
IMGTIME = 10000
IMG_ITI = (750, 1500)
BASELINETIME = 200
PUPTRIALTIME = 2500
PUP_ITI = (500, 1500)
FIXTIME = 2000
ATT_PUP_FIXTIME = 1500
ATT_PUP_POSTCUETIME = 2500

# SDK key. This is user-specific, so needs to be user defined.
ALEAKEY = "Contact Alea for an API key"
# Boolean that determines whether an animated calibration should be used. This
# is a friendly parrot; ideal for children.
ALEAANIMATEDCALIBRATION = False
# Alea offers their own specific type of logging with a specific output 
# location and file layout. PyGaze offers a different way that relies on 
# streaming data. Alea prefer their own way, which is why the default is set
# to that. The PyGaze way of logging produces files that are more similar to
# those of other trackers.
ALEALOGGING = False
