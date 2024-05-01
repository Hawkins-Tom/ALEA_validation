#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:06:23 2024

@author: tomhawkins
"""

import os
import math
from math import atan2, degrees
import numpy

#Files and folders
DIR = os.path.dirname(os.path.abspath('__file__'))
DATADIR = os.path.join(DIR, "data")
if not os.path.isdir(DATADIR):
    raise Exception("Could not find data at {}" .format(DATADIR))

#Image folder
IMGDIR = os.path.join(DIR, 'images')

#Data folder
DATADIR = os.path.join(DIR, 'data')

# Output folder to save all graphs and tables to 
OUTDIR = os.path.join(DIR, 'output')
# Specific output folders for the various tests
CALDIR = os.path.join(OUTDIR, 'calibration')
AOIDIR = os.path.join(OUTDIR, 'aoi')
PUPDIR = os.path.join(OUTDIR, 'pupil')
ATTDIR = os.path.join(OUTDIR, 'att_pupil')

#Conversion calculation for degree visual angle to pixels
def deg2px(v, d, px_per_cm):
    s_cm = 2 * d * numpy.tan(numpy.radians(v)/2.0)
    return s_cm*px_per_cm

def px2deg(s, d, px_per_cm):
    v = numpy.degrees(2 * numpy.arctan((s/px_per_cm) / (2*d)))
    return v

#Calculates distance between two points
def linedis(x1, x2, y1, y2):
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

#Screen information
DISPSIZE = (1440,900) #(1920, 1080)
SCREENSIZE = (42.0, 26.0)
PX_PER_CM = ((DISPSIZE[0]/SCREENSIZE[0]) + (DISPSIZE[1]/SCREENSIZE[1])) / 2.0
SCREENDIST = 67.0

# calculate the deg per pixel for the display 
DEG_PER_PX = degrees(atan2(.5 * SCREENSIZE[0], SCREENDIST)) / (.5 * DISPSIZE[1])


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

##### ANALYSIS constants

#Fixations
FIXTHRESHOLD_PX = deg2px(1, SCREENDIST, PX_PER_CM)

FIXMINDUR = 50 #milliseconds

#Minimum number of samples required for a fixation
ALEA_SAMPLE_MS = 1000/40
EYELINK_SAMPLE_MS = 1000/1000

ALEA_SAMPLE_MIN = round(FIXMINDUR/ ALEA_SAMPLE_MS, 1)

EYELINK_SAMPLE_MIN = round(FIXMINDUR/ EYELINK_SAMPLE_MS, 1)
